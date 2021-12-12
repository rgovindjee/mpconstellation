from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
from linearize_discretize import Discretizer
from control import *
from satellite_scale import SatelliteScale
import constants
import numpy as np
import random
from optimizer import Optimizer

# Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
# Initial position
r_init = np.array([6920.6596, 0, 0]) * 1000  # m
# Initial velocity
v_init = np.array([0, 7.58887, 0]) * 1000  # m/s
# Initial Mass
m_init = 12200  # kg
# Create satellite object
sat = Satellite(r_init, v_init, m_init)
# Create a scaling object
scale = SatelliteScale(sat=sat)
# Normalize system parameters (pg. 21)
const = scale.get_normalized_constants()

# Initial Thrust
T_tan_mag = 0.5  # Tangential thrust magnitude
c = ConstantTangentialThrustController([sat], T_tan_mag)
# Adjustable parameters
tf = 2
base_res = 400
# Use simulator to generate a reference (guess) trajectory
sim = Simulator(sats=[sat], controller=c, scale=scale,
                base_res=base_res, include_drag=False, include_J2=False)
sim.run(tf=tf)
# Create discretizer object with default arguments (no drag, no J2)
d = Discretizer(const, use_scipy_ZOH=False,
                include_drag=False, include_J2=False)
# Set up inputs
x_bar = sim.sim_data[sat.id]  # Guess trajectory from simulation
initial_sim = sim.sim_data[sat.id]
K = x_bar.shape[1]  # K = int(base_res*tf)
u_bar = Discretizer.extract_uk(
    x_bar, sim.sim_time[sat.id], c)  # Guess inputs
nu_bar = np.zeros((7, K))
f = Simulator.satellite_dynamics
# Run SCP:
N_max = 5
r_des = np.linalg.norm(x_bar[0:3, -1])

for n in range(N_max):
    opt_options = {'r_des': r_des}
    opt = Optimizer([x_bar], [u_bar], [nu_bar], tf, d, f, scale)
    opt.solve_OPT(input_options=opt_options)
    print(f"Iteration: {n}")
    x_bar = opt.get_solved_trajectory(0)
    u_bar = opt.get_solved_u(0)
    nu_bar = opt.get_solved_nu(0)
    tf = opt.get_solved_tf(0)

# Plot converged result
x_final = scale.redim_state(x_bar)
plot_orbit_3D(trajectories=[x_final],
              references=[scale.redim_state(initial_sim)],
              use_mayavi=True)

# Simulate nonlinear trajectory for more orbits using control outputs
tf_sim = 5
tf_u = tf
c_opt = SequenceController(u=u_bar, tf_u=tf_u, tf_sim=tf_sim)
sim = Simulator(sats=[sat], controller=c_opt, scale=scale,
                base_res=base_res, include_drag=False, include_J2=False)
sim.run(tf=tf_sim)
x_forward = sim.sim_data[sat.id]  # Guess trajectory from simulation
plot_orbit_3D(trajectories=[x_final],
              references=[scale.redim_state(x_forward)],
              use_mayavi=True)
