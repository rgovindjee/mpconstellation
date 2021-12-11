from numpy.core.numeric import indices
from pyomo.core.base.expression import ScalarExpression
import pyomo.environ as pyo
import numpy as np
from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
from satellite_scale import SatelliteScale
from linearize_discretize import Discretizer
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, x_bar, u_bar, nu_bar, tf, d, f, scale):
        """
            Arguments:
            x_bar: A list of N elements (N satellites). Each element is a numpy array of shape
                (7, K) reference state trajectories
            u_bar: A list of N elements. Each element is a numpy array of shape
                (3, K) reference thrust trajectories
            nu_bar: A list of N elements. Each element is a numpy array of shape
                (7, K) virtual control vectors
            tf: Scalar
            d: Discretizer object
            f: Satellite dynamics function/method
            scale: A SatelliteScale object
        """

        self.x_bar = x_bar
        self.u_bar = u_bar
        self.nu_bar = nu_bar
        self.tf = tf
        self.d = d
        self.f = f
        self.scale = scale
        self.const = scale.get_normalized_constants()
        self._N = len(x_bar)
        self._K = x_bar[0].shape[1]

    @staticmethod
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

    def get_constraint_terms(self):
        """
        Get constraint terms for N satellites

        Returns:
            dict containing the following keys:
                rbar_hat:       N-element list of 2D numpy arrays (3, K-1)
                ubar_hat:       N-element list of 2D numpy arrays (3, K-1)
                rf_hat:         N-element list of 1D numpy vector (3, )
                Vc:             N-element list of scalars
                DrVc:           N-element list of 1D numpy vector (3, )
                DrVc_rbar:      N-element list of scalars
                Vt:             N-element list of scalars
                DrVt_DvVt:      N-element list of 1D numpy vector (6, )
                DrVt_DvVt_bar:  N-element list of scalars
                Vr:             N-element list of scalars
                DrVr_DvVr:      N-element list of 1D numpy vector (6, )
                DrVr_DvVr_bar:  N-element list of scalars
                Vn:             N-element list of scalars
                DrVn_DvVn:      N-element list of 1D numpy vector (6, )
                DrVn_DvVn_bar:  N-element list of scalars
        """
        # Preallocate dict
        keys = ['rbar_hat', 'ubar_hat', 'rf_hat', 'Vc', 'DrVc',
                                'DrVc_rbar', 'Vt', 'DrVt_DvVt', 'DrVt_DvVt_bar',
                                'Vr', 'DrVr_DvVr', 'DrVr_DvVr_bar', 'Vn', 'DrVn_DvVn',
                                'DrVn_DvVn_bar']
        output = {k: [] for k in keys}

        I = np.eye(3) # Identity matrix
        for i in range(self._N):
            # Get RTN unit vectors of final position
            r_bar_K = self.x_bar[i][0:3,-1]
            r_final_norm = np.linalg.norm(r_bar_K)
            v_bar_K = self.x_bar[i][3:6,-1]
            rv_bar_K = np.concatenate([r_bar_K, v_bar_K])
            h_K = np.cross(r_bar_K, v_bar_K)
            r_hat_K = r_bar_K/r_final_norm
            h_hat_K = h_K/np.linalg.norm(h_K)
            t_hat_K = np.cross(h_hat_K, r_hat_K)
            # Partial derivatives of unit vectors
            Dr_h_hat = ((np.linalg.norm(h_K)**-1 * I) - (np.linalg.norm(h_K)**-3 * np.outer(h_K, h_K))) @ (-self.skew(v_bar_K))
            Dv_h_hat = (np.linalg.norm(h_K)**-1 * I) - (np.linalg.norm(h_K)**-3 * np.outer(h_K, h_K)) @ (self.skew(r_bar_K))
            Dr_r_hat = (r_final_norm**-1 * I) - (r_final_norm**-3 * np.outer(r_bar_K, r_bar_K))
            Dr_t_hat = (-self.skew(r_hat_K) @ Dr_h_hat) + (self.skew(h_hat_K) @ Dr_r_hat)
            Dv_t_hat = -self.skew(r_hat_K) @ Dv_h_hat

            # Constraints that apply for k = 1...K-1 -> 1, 2, 3 -> 0, 1, 2
            # Distance constraint terms
            r_bar = self.x_bar[i][0:3,:-1]
            output['rbar_hat'].append(r_bar/np.linalg.norm(r_bar, axis=0))
            # Thrust constraint terms
            u_bar = self.u_bar[i]
            output['ubar_hat'].append(u_bar/np.linalg.norm(u_bar, axis=0))


            # Constraints that apply for the final point K
            # Final position constraint terms
            output['rf_hat'].append(r_hat_K)
            # Tangential velocity constraint terms
            output['Vc'].append(np.sqrt(self.const.MU/r_final_norm))
            DrVc = (-1/2)*(self.const.MU**0.5)*(r_final_norm**(-5/2))*r_bar_K
            output['DrVc'].append(DrVc)
            output['DrVc_rbar'].append(np.dot(DrVc, r_bar_K))
            output['Vt'].append(np.dot(v_bar_K, t_hat_K))
            DrVt = np.dot(v_bar_K, Dr_t_hat)
            DvVt = np.dot(t_hat_K, I) + np.dot(v_bar_K, Dv_t_hat)
            DrVt_DvVt = np.concatenate([DrVt, DvVt])
            output['DrVt_DvVt'].append(DrVt_DvVt)
            output['DrVt_DvVt_bar'].append(np.dot(DrVt_DvVt, rv_bar_K))
            # Radial velocity constraint terms
            output['Vr'].append(np.dot(v_bar_K, r_hat_K))
            DrVr = np.dot(v_bar_K, Dr_r_hat)
            DvVr = np.dot(r_hat_K, I)
            DrVr_DvVr = np.concatenate([DrVr, DvVr])
            output['DrVr_DvVr'].append(DrVr_DvVr)
            output['DrVr_DvVr_bar'].append(np.dot(DrVr_DvVr, rv_bar_K))
            # Normal velocity constraint terms
            output['Vn'].append(np.dot(v_bar_K, h_hat_K))
            DrVn = np.dot(v_bar_K, Dr_h_hat)
            DvVn = np.dot(h_hat_K, I) + np.dot(v_bar_K, Dv_h_hat)
            DrVn_DvVn = np.concatenate([DrVn, DvVn])
            output['DrVn_DvVn'].append(DrVn_DvVn)
            output['DrVn_DvVn_bar'].append(np.dot(DrVn_DvVn, rv_bar_K))
        return output

    def init_options(self, options):
        """
        Initializes options dictionary for solver by merging with default options
        Args:
            options: dictionary of options, may be the empty dict
        """
        default = { 'min_mass':0.5,
                    'u_lim':[0.1, 2],
                    'r_lim':[0, 100],
                    'r_des':10000,
                    'eps_max':100,
                    'tf_max':5,
                    'w_nu':10000}
        merged = {**default, **options}
        return merged

    def solve_OPT(self, input_options={}):
        """
        Transcribes and solves the OPT problem, for N satellites

        Args:
            options: dictionary of options
                min_mass = 0.5  # Mass limit, normalized
                u_lim = [0.1, 2] # Thrust limit, normalized
                r_lim = [0, 100] # Bound on radial distance (altitude), normalized
                r_des = 10000 # Desired final orbital altitude, normalized
                eps_max = 100 # Upper bound for slack variables epsilon, tunable
                tf_max = 5 # Upper bound for final time, tunable
                w_nu = 10000 # Penalty for virtual control
        """
        options = self.init_options(options=input_options)

        # First, we must transcribe the OPT:
        # Discretize and linearize dynamics for all satellites
        A_k = []
        B_kp = []
        B_kn = []
        Sigma_k = []
        xi_k = []
        for i in range(len(self.x_bar)):
            A_k_i, B_kp_i, B_kn_i, Sigma_k_i, xi_k_i = self.d.discretize(self.f, self.x_bar[i], self.u_bar[i], self.tf)
            A_k.append(A_k_i)
            B_kp.append(B_kp_i)
            B_kn.append(B_kn_i)
            Sigma_k.append(Sigma_k_i)
            xi_k.append(xi_k_i)
        # Create linearized, discretized coefficients used in constraints
        cons_terms = self.get_constraint_terms()

        # Formulate and solve Pyomo problem
        model = pyo.ConcreteModel()

        # Get indices
        model.K = self._K # Number of time points
        model.N = self._N # Number of satellites
        model.xdim = 7
        model.udim = 3
        model.kIDX = pyo.Set(initialize=range(self._K))
        model.sIDX = pyo.Set(initialize=range(self._N))
        model.xIDX = pyo.Set(initialize=range(model.xdim))
        model.uIDX = pyo.Set(initialize=range(model.udim))

        # Create state and input variables trajectory:
        model.x = pyo.Var(model.sIDX, model.xIDX, model.kIDX)
        model.u = pyo.Var(model.sIDX, model.uIDX, model.kIDX)
        model.nu = pyo.Var(model.sIDX, model.xIDX, model.kIDX) # Virtual controls
        model.t = pyo.Var(model.sIDX, model.xIDX, model.kIDX) # Slack variable for L1 minimization of nu

        # Slack for final radial height
        model.eps_r = pyo.Var()
        # Slack Variables for Circularization Constraints
        model.eps_vr = pyo.Var()
        model.eps_vn = pyo.Var()
        model.eps_vt = pyo.Var()

        # For minimum time problems
        model.tf = pyo.Var()

        # Add linearized/discretized matrices to model
        model.A_k = A_k
        model.B_kp = B_kp
        model.B_kn = B_kn
        model.Sigma_k = Sigma_k
        model.xi_k = xi_k

        # Store constraint terms
        model.cons_terms = cons_terms

        # Objective: Minimize Time For Orbit Raising and Circularization
        def minimize_time(model):
            cost = (model.tf + options['w_nu']*sum(model.t[s,i,k] for k in model.kIDX for i in model.xIDX for s in model.sIDX))
            return cost

        def dynamics_const_rule(model, s, i, k):
            try:
                if k+1 >= model.K:
                    return pyo.Constraint.Skip
                else:
                    return (model.x[s, i, k+1]
                            - (sum(model.A_k[s][k, i, j]  * model.x[s, j, k] for j in model.xIDX)
                                + sum(model.B_kn[s][k, i, j] * model.u[s, j, k] for j in model.uIDX)
                                + sum(model.B_kp[s][k, i, j] * model.u[s, j, k+1] for j in model.uIDX)
                                + (model.Sigma_k[s][i, k] * model.tf)
                                + model.xi_k[s][i, k]
                                + model.nu[s,i,k])
                            == 0.0)
            except Exception as e:
                print(f"s={s}, i={i}, k={k}")
                raise e

        def initial_state_rule(model, s, i):
            return model.x[s, i, 0] == self.x_bar[s][i,0]

        def initial_thrust_rule(model, s, i):
            return model.u[s, i, 0] == self.u_bar[s][i,0]

        def final_mass_rule(model, s):
            return model.x[s, i, model.K-1] >= options['min_mass']

        # Set cost function
        model.cost = pyo.Objective(rule=minimize_time, sense=pyo.minimize)
        # Initialize States
        model.init_states = pyo.Constraint(model.sIDX, model.xIDX, rule=initial_state_rule)
        # Initialize Inputs
        model.init_inputs = pyo.Constraint(model.sIDX, model.uIDX, rule=initial_thrust_rule)
        # Linearized Dynamics Constraints
        model.dynamics = pyo.Constraint(model.sIDX, model.xIDX, model.kIDX, rule=dynamics_const_rule)
        # Final mass above minimum constraint
        model.mass_final = pyo.Constraint(model.sIDX, rule=final_mass_rule)

        # Thrust Contstraints
        def min_thrust_rule(model, s, k):
            if k+1 >= model.K:
                return pyo.Constraint.Skip
            else:
                thrusts = [model.cons_terms['ubar_hat'][s][i][k] * model.u[s,i,k] for i in model.uIDX]
                return (sum(thrusts) >= options['u_lim'][0])

        model.thrust_min = pyo.Constraint(model.sIDX, model.kIDX, rule=min_thrust_rule)

        model.thrust_max = pyo.Constraint(model.sIDX, model.kIDX, rule=lambda model, s,
                                        k: model.u[s,0,k]**2 + model.u[s,1,k]**2 + model.u[s,2,k]**2 <= options['u_lim'][1]**2
                                        if k < model.K else pyo.Constraint.Skip)

        # Constraints on the bounds of the trajectory radial distance
        def min_dist_rule(model, s, k):
            if k+1 >= model.K:
                return pyo.Constraint.Skip
            else:
                return (sum(model.cons_terms['rbar_hat'][s][i][k] * model.x[s,i,k] for i in range(3))
                    >= options['r_lim'][0])

        model.radial_min = pyo.Constraint(model.sIDX, model.kIDX, rule = min_dist_rule)

        model.radial_max = pyo.Constraint(model.sIDX, model.kIDX, rule=lambda model, s,
                                        k: model.x[s,0,k]**2 + model.x[s,1,k]**2 + model.x[s,2,k]**2 <= options['r_lim'][1]**2
                                        if k < model.K else pyo.Constraint.Skip)

        # Constraints on the final radial distance
        def min_final_dist_rule(model, s):
            radial_dists = [model.cons_terms['rf_hat'][s][i] * model.x[s,i,model.K-1] for i in range(3)]
            print("shape of rf_hat:")
            print(model.cons_terms['rf_hat'][s].shape)
            print([model.cons_terms['rf_hat'][s][i] for i in range(3)])
            return (sum(radial_dists) >= (options['r_des'] - model.eps_r))

        model.radial_final_min = pyo.Constraint(model.sIDX, rule=min_final_dist_rule)
        model.radial_final_max = pyo.Constraint(model.sIDX, rule=lambda model, s: model.x[s, 0, model.K-1]**2 + model.x[s, 1, model.K-1]**2 + model.x[s, 2, model.K-1]**2  <= (options['r_des'] + model.eps_r)**2)

        # Radial Velocity Constraints
        def max_radial_vel_rule(model, s):
            return (model.cons_terms['Vr'][s]
                    + sum(model.cons_terms['DrVr_DvVr'][s][i] * model.x[s,i,model.K-1] for i in range(6))
                    - model.cons_terms['DrVr_DvVr_bar'][s]
                    <= model.eps_vr)

        def min_radial_vel_rule(model, s):
            return (-1*(model.cons_terms['Vr'][s]
                        + sum(model.cons_terms['DrVr_DvVr'][s][i] * model.x[s,i,model.K-1] for i in range(6))
                        - model.cons_terms['DrVr_DvVr_bar'][s])
                    >= -1*model.eps_vr)

        model.vr_max = pyo.Constraint(model.sIDX, rule=max_radial_vel_rule)
        model.vr_min = pyo.Constraint(model.sIDX, rule=min_radial_vel_rule)

        # Normal Velocity Constraints
        def max_normal_vel_rule(model, s):
            return (model.cons_terms['Vn'][s]
                    + sum(model.cons_terms['DrVn_DvVn'][s][i] * model.x[s,i,model.K-1] for i in range(6))
                    - model.cons_terms['DrVn_DvVn_bar'][s]
                    <= model.eps_vn)

        def min_normal_vel_rule(model, s):
            return (-1*(model.cons_terms['Vn'][s]
                        + sum(model.cons_terms['DrVn_DvVn'][s][i] * model.x[s,i,model.K-1] for i in range(6))
                        - model.cons_terms['DrVn_DvVn_bar'][s])
                    >= -1*model.eps_vn)

        model.vn_max = pyo.Constraint(model.sIDX, rule=max_normal_vel_rule)
        model.vn_min = pyo.Constraint(model.sIDX, rule=min_normal_vel_rule)

        # Tangential Velocity Constraints
        def max_tan_vel_rule(model, s):
            return (model.cons_terms['Vc'][s]
                    + sum(model.cons_terms['DrVc'][s][i]*model.x[s,i,model.K-1] for i in range(3))
                    - model.cons_terms['DrVc_rbar'][s]
                    - model.eps_vt
                    - model.cons_terms['Vt'][s]
                    - sum(model.cons_terms['DrVt_DvVt'][s][i] * model.x[s,i,model.K-1] for i in range(6))
                    + model.cons_terms['DrVt_DvVt_bar'][s]
                    <= 0.0)

        def min_tan_vel_rule(model, s):
            return (model.cons_terms['Vt'][s]
                    + sum(model.cons_terms['DrVt_DvVt'][s][i] * model.x[s,i,model.K-1] for i in range(6))
                    - model.cons_terms['DrVt_DvVt_bar'][s]
                    - model.cons_terms['Vc'][s]
                    - sum(model.cons_terms['DrVc'][s][i]*model.x[s,i,model.K-1] for i in range(3))
                    + model.cons_terms['DrVc_rbar'][s]
                    - model.eps_vt
                    <= 0.0)

        model.vt_max = pyo.Constraint(model.sIDX, rule=max_tan_vel_rule)
        model.vt_min = pyo.Constraint(model.sIDX, rule=min_tan_vel_rule)

        # L1-norm minimization slack variable constraints
        def pos_t_rule(model, s, i, k):
            return model.nu[s, i, k] <= model.t[s, i, k]
        def neg_t_rule(model, s, i, k):
            return -1*(model.nu[s, i, k]) <= model.t[s, i, k]

        model.t_pos = pyo.Constraint(model.sIDX, model.xIDX, model.kIDX, rule = pos_t_rule)
        model.t_neg = pyo.Constraint(model.sIDX, model.xIDX, model.kIDX, rule = neg_t_rule)

        # tf needs to be positive but upper bound can be changed
        model.tf_limits = pyo.Constraint(expr=(0, model.tf, options['tf_max']))

        # Bounds for the slack variable used in the final distance constraints
        model.eps_r_limits = pyo.Constraint(expr=(0, model.eps_r, options['eps_max']))

        # Bounds for the slack variables that are used in the circularization constraints
        model.eps_vr_limits = pyo.Constraint(expr=(0, model.eps_vr, options['eps_max']))
        model.eps_vn_limits = pyo.Constraint(expr=(0, model.eps_vn, options['eps_max']))
        model.eps_vt_limits = pyo.Constraint(expr=(0, model.eps_vt, options['eps_max']))

        # Solve model
        model.dual = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        solver = pyo.SolverFactory('ipopt')
        # solver.options['max_iter'] = 1000
        results = solver.solve(model, tee=False)

        #xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
        #uOpt = np.asarray([[model.u[j,t]() for j in model.uIDX] for t in model.tIDX]).T
        #JOpt = model.cost()

        #tfOpt = pyo.value(model.tf)

        #print(f'eps_r: {pyo.value(model.eps_r)}')
        #print(f'tf: {pyo.value(model.tf)}')
        self.model = model
