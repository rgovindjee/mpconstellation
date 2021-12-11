import pyomo.environ as pyo
import numpy as np
from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
from satellite_scale import SatelliteScale
from linearize_discretize import Discretizer
import matplotlib.pyplot as plt


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def plot_normalized_thrust(u, T0):
    u = u/T0
    fig, ax = plt.subplots()
    time = np.linspace(0,1,len(u))
    ax.plot(time, u[:,0], label='x')
    ax.plot(time, u[:,1], label='y')
    ax.plot(time, u[:,2], label='z')
    ax.set_title('Normalized Thrust Commands')
    plt.show()


def solve_optimal_control(N, tf, x0, u0, A_k, B_kp, B_kn, Sigma_k, xi_k, x_lim, u_lim, x_f, ref_pos_norm, v_s, v_m, r_f, v_f):
    """
    Solves optimal control problem with the given constraints 
    """
    model = pyo.ConcreteModel()
    model.N = N
    model.nx = np.size(x0)
    model.nu = np.size(u0)

    # Length of optimization problem:
    model.tIDX = pyo.Set(initialize=range(0, N+1))
    model.xIDX = pyo.Set(initialize=range(0, model.nx))
    model.uIDX = pyo.Set(initialize=range(0, model.nu))

    # Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.tIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX)
    
    # Slack for final radial height
    model.eps_r = pyo.Var()

    # For minimum time problems 
    model.tf = pyo.Var()

    # Add linearized/discretized matrices to model
    model.A_k = A_k
    model.B_kp = B_kp
    model.B_kn = B_kn
    model.Sigma_k = Sigma_k
    model.xi_k = xi_k

    # Slack Variables for Circularization Constraints
    model.eps_vr = pyo.Var()
    model.eps_vn = pyo.Var()
    model.eps_vt = pyo.Var()

    # Needed non-linear mass dot equation
    non_zero = 1e-8

    # Objective: Minimize Time For Orbit Raising and Circularization
    def minimize_time(model):
        cost = model.tf**2
        return cost
    
    def dynamics_const_rule(model, i, t):
        return model.x[i, t+1] - (sum(model.A_k[i, j]  * model.x[j, t] for j in model.xIDX)
                               +  sum(model.B_kp[i, j] * model.u[j, t] for j in model.uIDX)
                               +  sum(model.B_kn[i, j] * model.u[j, t] for j in model.uIDX)
                               +  sum(model.Sigma_k[j] * model.tf for j in model.xIDX)
                               +  sum(model.xi_k[j] for j in model.xIDX)) == 0.0 if t < model.N else pyo.Constraint.Skip

    # Set cost function
    model.cost = pyo.Objective(rule=minimize_time, sense=pyo.minimize)

    # Initialize States
    model.init_states = pyo.Constraint(model.xIDX, rule=lambda model, i: model.x[i, 0] == x0[i])

    # Initialize Inputs
    model.init_inputs = pyo.Constraint(model.uIDX, rule=lambda model, i: model.u[i, 0] == u0[i])

    # Linearized Dynamics Constraints
    model.dynamics = pyo.Constraint(model.xIDX, model.tIDX, rule=dynamics_const_rule)
    model.mass_final = pyo.Constraint(expr=(model.x[6,N] >= .9))

    # Thrust Contstraints
    model.thrust_max = pyo.Constraint(model.tIDX, rule=lambda model, 
                                      t: model.u[0,t]**2 + model.u[1,t]**2 + model.u[2,t]**2 <= u_lim[1]**2 
                                      if t < N else pyo.Constraint.Skip)

    # Constraints on the bounds of the trajectory radial distance
    model.radial_min = pyo.Constraint(model.tIDX, rule=lambda model, 
                                      t: ref_pos_norm[0,t] * model.x[0,t] + ref_pos_norm[1,t] * model.x[1,t] + ref_pos_norm[2,t] * model.x[2,t] >= x_lim[0]
                                      if t < N else pyo.Constraint.Skip)

    model.radial_max = pyo.Constraint(model.tIDX, rule=lambda model, 
                                      t: model.x[0,t]**2 + model.x[1,t]**2 + model.x[2,t]**2 <= x_lim[1]**2 
                                      if t < N else pyo.Constraint.Skip)

    # Constraints on the final radial distance
    model.radial_final_min = pyo.Constraint(model.tIDX, rule=lambda model, 
                                      t: r_final[0] * model.x[0,N] + r_final[1] * model.x[1,N] + r_final[2] * model.x[2,N] >= (x_f[0] - model.eps_r)**2
                                      if t < N else pyo.Constraint.Skip)
    model.radial_final_max = pyo.Constraint(expr=model.x[0, N]**2 + model.x[1, N]**2 + model.x[2, N]**2  <= (x_f[0] + model.eps_r)**2)

    # Radial Velocity Constraints
    model.vr_max = pyo.Constraint(expr=(v_s[0] + (
                                      v_m[0][0] * (model.x[0, N] - r_f[0])
                                    + v_m[0][1] * (model.x[1, N] - r_f[1])
                                    + v_m[0][2] * (model.x[2, N] - r_f[2])
                                    + v_m[0][3] * (model.x[3, N] - v_f[0])
                                    + v_m[0][4] * (model.x[4, N] - v_f[1])
                                    + v_m[0][5] * (model.x[5, N] - v_f[2]))) <= model.eps_vr)

    model.vr_min = pyo.Constraint(expr=-(v_s[0] + (
                                      v_m[0][0] * (model.x[0, N] - r_f[0])
                                    + v_m[0][1] * (model.x[1, N] - r_f[1])
                                    + v_m[0][2] * (model.x[2, N] - r_f[2])
                                    + v_m[0][3] * (model.x[3, N] - v_f[0])
                                    + v_m[0][4] * (model.x[4, N] - v_f[1])
                                    + v_m[0][5] * (model.x[5, N] - v_f[2]))) >= -model.eps_vr)

    # Normal Velocity Constraints
    model.vn_max = pyo.Constraint(expr=v_s[1] + (
                                      v_m[1][0] * (model.x[0, N] - r_f[0])
                                    + v_m[1][1] * (model.x[1, N] - r_f[1])
                                    + v_m[1][2] * (model.x[2, N] - r_f[2])
                                    + v_m[1][3] * (model.x[3, N] - v_f[0])
                                    + v_m[1][4] * (model.x[4, N] - v_f[1])
                                    + v_m[1][5] * (model.x[5, N] - v_f[2])) <= model.eps_vn)

    model.vn_min = pyo.Constraint(expr=-(v_s[1] + (
                                      v_m[1][0] * (model.x[0, N] - r_f[0])
                                    + v_m[1][1] * (model.x[1, N] - r_f[1])
                                    + v_m[1][2] * (model.x[2, N] - r_f[2])
                                    + v_m[1][3] * (model.x[3, N] - v_f[0])
                                    + v_m[1][4] * (model.x[4, N] - v_f[1])
                                    + v_m[1][5] * (model.x[5, N] - v_f[2]))) >= -model.eps_vn)

    # Tangential Velocity Constraints
    model.vt_max = pyo.Constraint(expr=v_s[2] + (
                                      v_m[2][0] * (model.x[0, N] - r_f[0])
                                    + v_m[2][1] * (model.x[1, N] - r_f[1])
                                    + v_m[2][2] * (model.x[2, N] - r_f[2])
                                    + v_m[2][3] * (model.x[3, N] - v_f[0])
                                    + v_m[2][4] * (model.x[4, N] - v_f[1])
                                    + v_m[2][5] * (model.x[5, N] - v_f[2])) 
                                    <= 
                                      v_s[3] + ( 
                                      v_m[3][0] * (model.x[0, N] - r_f[0])
                                    + v_m[3][1] * (model.x[1, N] - r_f[1])
                                    + v_m[3][2] * (model.x[2, N] - r_f[2]) + model.eps_vt))

    model.vt_min = pyo.Constraint(expr=v_s[3] + ( 
                                       v_m[3][0] * (model.x[0, N] - r_f[0])
                                    +  v_m[3][1] * (model.x[1, N] - r_f[1])
                                    +  v_m[3][2] * (model.x[2, N] - r_f[2]) - model.eps_vt)
                                      <= 
                                      v_s[2] + (
                                      v_m[2][0] * (model.x[0, N] - r_f[0])
                                    + v_m[2][1] * (model.x[1, N] - r_f[1])
                                    + v_m[2][2] * (model.x[2, N] - r_f[2])
                                    + v_m[2][3] * (model.x[3, N] - v_f[0])
                                    + v_m[2][4] * (model.x[4, N] - v_f[1])
                                    + v_m[2][5] * (model.x[5, N] - v_f[2]))) 

    # tf needs to be positive but upper bound can be changed
    model.tf_limits = pyo.Constraint(expr=(0, model.tf, 10.0))

    # Bounds for the slack variable used in the final distance constraints
    model.eps_r_limits = pyo.Constraint(expr=(0, model.eps_r, 1000))

    # Bounds for the slack variables that are used in the circularization constraints
    model.eps_vr_limits = pyo.Constraint(expr=(0, model.eps_vr, 1000))
    model.eps_vn_limits = pyo.Constraint(expr=(0, model.eps_vn, 1000))
    model.eps_vt_limits = pyo.Constraint(expr=(0, model.eps_vt, 1000))

    model.dual = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    solver = pyo.SolverFactory('ipopt')
    # solver.options['max_iter'] = 1000
    results = solver.solve(model, tee=False)

    xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
    uOpt = np.asarray([[model.u[j,t]() for j in model.uIDX] for t in model.tIDX]).T
    JOpt = model.cost()

    tfOpt = pyo.value(model.tf)

    print(f'eps_r: {pyo.value(model.eps_r)}')
    print(f'tf: {pyo.value(model.tf)}')

    return [model, xOpt, uOpt, JOpt, tfOpt]


# Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
# Initial position
sat_position = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
# Initial velocity
sat_velocity = np.array([4.6921, 4.9848, -3.2752]) * 1000  # m/s
# Initial Thrust
sat_thrust = np.array([0, 0, 0])  # N
# Initial Mass
sat_mass = 12200  # kg
m0 = sat_mass

# Run Simulator
sat = Satellite(sat_position, sat_velocity, m0)
scale = SatelliteScale(sat = sat)
sim = Simulator(sats=[sat], scale=scale)  # Use default controller

# Constants
r0 = np.linalg.norm(sat_position)
s0 = 2*np.pi*np.sqrt(r0**3/MU_EARTH)
v0 = r0/s0
a0 = r0/s0**2
m0 = sat_mass
T0 = m0*r0/s0**2
mu0 = r0**3/s0**2
const = Constants(MU=MU_EARTH/mu0, R_E=R_EARTH/r0, J2=J2, G0=G0/a0, ISP=ISP/s0, S=S/r0**2, R0=r0, RHO=s0)

d = Discretizer(const, use_scipy_ZOH = True)
# Set up inputs
x = sat.get_state_vector()
x = np.column_stack([x, x])
u = np.array([0, 0, 0])
u = np.column_stack([u, u])
tf = 1
K = 2
f = Simulator.satellite_dynamics
A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)

# Initial States and Inputs
x0 = np.concatenate([sat_position/r0, sat_velocity/v0, np.array([sat_mass/m0])])
u0 = sat_thrust / T0

# State Constraints (normalized): [r_lower, r_upper]
x_lim = [0.90, 1.1]

# Input Constraints (normalized): [t_lower, t_upper]
thrust_max = 500 # N
u_lim = [0, thrust_max/T0]

# Final State Constraints (normalized): [r_des]
x_f = [1.01]

sim.run(tf=1)
ref_pos = sim.sim_data[sat.id][:3, :]
ref_vel = sim.sim_data[sat.id][3:6,:]

ref_pos_norm = np.zeros((3,100))

for t in range(len(ref_pos_norm[0])):
    ref_pos_norm[:,t] = ref_pos[:,t] / np.linalg.norm(ref_pos[:,t])

# Begin formulating parts of the circularization constraints
r_final = ref_pos[:, -1]
r_final_norm = np.linalg.norm(r_final)
v_final = ref_vel[:, -1]
h_final = np.cross(r_final, v_final)
I = np.eye(3)

r_skew = skew(r_final)
v_skew = skew(v_final)

# Final Radial Speed
r_hat = r_final / r_final_norm
vr = np.dot(v_final, r_hat)
Dr_r_hat = r_final_norm**-1 * I - r_final_norm**-3 * r_final.T * r_final
Dr_vr = np.dot(v_final, Dr_r_hat)
Dv_vr = np.dot(r_hat, I)
vr_matrix = np.append(Dr_vr, Dv_vr)

# Final Normal Speed
h_hat = h_final / np.linalg.norm(h_final)
vn = np.dot(v_final, h_hat)
Dr_h_hat = np.linalg.norm(h_final)**-1 * I - np.linalg.norm(h_final)**-3 * h_final * h_final * -v_skew
Dv_h_hat = np.linalg.norm(h_final)**-1 * I - np.linalg.norm(h_final)**-3 * h_final * h_final * r_skew
Dr_vn = np.dot(v_final, Dr_h_hat)
Dv_vn = np.dot(h_hat, I) + np.dot(v_final, Dv_h_hat)
vn_matrix = np.append(Dr_vn, Dv_vn)

# Final Tangential Speed
t_hat = np.cross(h_hat, r_hat)
vt = np.dot(v_final, t_hat)
Dr_t_hat = -skew(r_hat) * Dr_h_hat + skew(h_hat) * Dr_r_hat
Dv_t_hat = -skew(r_hat) * Dv_h_hat
Dr_vt = np.dot(v_final, Dr_t_hat)
Dv_vt = np.dot(t_hat, I) + np.dot(v_final, Dv_t_hat)
vt_matrix = np.append(Dr_vt, Dv_vt)

# Circular Speed
vc = np.sqrt(const.MU/ np.linalg.norm(ref_pos))
dr_vc = -.5 * const.MU**.5 * r_final_norm**(-5/2) * r_final

v_speeds = np.array([vr, vn, vt, vc])
v_matrices = np.array([vr_matrix, vn_matrix, vt_matrix, dr_vc])

N = 100
[model, xOpt, uOpt, JOpt, tfOpt] = solve_optimal_control(N, tf, x0, u0, A_k[0], B_kp[0], B_kn[0], Sigma_k[0], xi_k[0], x_lim, u_lim, x_f, ref_pos_norm, v_speeds, v_matrices, r_final, v_final)

sim.run(tfOpt)
ref = scale.redim_state(sim.sim_data[sat.id])
traj = scale.redim_state(xOpt)
plot_orbit_3D(trajectories=[traj], references=[ref], use_mayavi = False)
# plot_normalized_thrust(np.where(uOpt.T == None, 0, uOpt.T), T0)
