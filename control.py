import numpy as np
import simulator
from satellite_scale import SatelliteScale
from linearize_discretize import Discretizer
from optimizer import Optimizer
from sim_plotter import *

class Controller:
    """
    Controllers track one or more satellites and produce thrust commands for each satellite under control.
    """
    def __init__(self, sats=[]):
        """
        Arguments:
            sats: list of Satellite objects
        """
        self.sats = sats
        self.sat_ids = set(s.id for s in sats)

    def get_u_func(sat_id=None):
        """
        Arguments:
            sat_id: id of satellite to return output for.
        Returns an open-loop control function u(t) for a given satellite.
        """
        #Satellite id irrelevant for base controller.
        zero_thrust = np.array([0., 0., 0.])
        u = lambda x, tau: zero_thrust
        return u

    def update(self):
        """
        Recalculate according to current satellite states if needed
        """
        pass

class ConstantThrustController(Controller):
    def __init__(self, sats=[], thrust=np.array([1., 1., 1.])):
        """
        Arguments:
            sats: list of Satellite objects
            thrust: 3-element array of thrust in x, y, z direction
        """
        super().__init__(sats)
        self.thrust = thrust

    def get_u_func(self, sat_id=None):
        """
        Arguments:
            sat_id: id of satellite to return output for.
        """
        u = lambda x, tau: self.thrust
        return u

class ConstantTangentialThrustController(Controller):
    def __init__(self, sats=[], tangential_thrust=1):
        """
        Arguments:
            sats: list of Satellite objects
            tangential_thrust: scalar, magnitude of tangential thrust
        """
        super().__init__(sats)
        self.tangential_thrust = tangential_thrust


    def compute_rotation(self, x):
        """
        Arguments:
            x: State vector of satellite
        Returns:
            3x3 rotation matrix from the RTN frame to ECI frame
        """
        r = x[0:3]
        v = x[3:6]
        r_hat = r/np.linalg.norm(r)
        h = np.cross(r, v)
        h_hat = h/np.linalg.norm(h)
        t_hat = np.cross(h_hat, r_hat)
        return np.column_stack([r_hat, t_hat, h_hat])


    def get_u_func(self, sat_id=None):
        u = lambda x, tau: self.compute_rotation(x) @ np.array([0,self.tangential_thrust,0])
        return u

class SequenceController(Controller):
    """
    Applies a sequence u over the given range, 0 otherwise. Linearly interpolates between samples
    """
    def __init__(self, sats=[], u=np.array([]), tf_u=1, tf_sim=1):
        """
        Arguments:
            sats: list of Satellite objects
            thrust: 3-element array of thrust in x, y, z direction
            tf_sim: tf for the simulation that is to be run
            tf_u: tf for which the inputs u were generated
        """
        super().__init__(sats)
        # Calculate the normalized time tau where the simulation inputs end, if applicable
        # (tau in simulation time)
        self.end_tau = tf_u / tf_sim
        self.u = u

    def u_FOH(self, tau):
        """
        First order hold interpolation of a signal u

        Args:
            tau: Current time tau
            u: n x K matrix of discrete inputs

        Returns:
            Interpolated u at time tau, u(tau)
        """
        if tau == 1:
            return self.u[:,-1]
        else:
            K = self.u.shape[1] # Number of points for discretization
            dtau = 1/(K-1) # Length of each interval in tau units
            k = int(tau // dtau) # lower index of interval to interpolate in
            tau_k = k/(K-1) # left bound of interval
            tau_kp1 = (k+1)/(K-1) # right bound of interval
            lambda_kn = (tau_kp1 - tau)/(tau_kp1 - tau_k)
            lambda_kp = (tau - tau_k)/(tau_kp1 - tau_k)
            return lambda_kn*self.u[:, k] + lambda_kp*self.u[:, k+1]

    def get_u_func(self, sat_id=None):
        """
        Arguments:
            sat_id: id of satellite to return output for.
        """
        def u(x, tau):
            if tau <= self.end_tau:
                t = tau/self.end_tau
                return self.u_FOH(t)
                ## Calculate nearest u index (3xK)
                #u_len = self.u.shape[1]
                #u_index = int((tau/self.end_tau) * (u_len-1))
                #return self.u[:, u_index]
            else:
                zero_thrust = np.array([0., 0., 0.])
                return zero_thrust
        return u

class OptimalController(Controller):
    def __init__(self, sats=[], objective=None, base_res=100, tf_horizon=1,
                 tf_interval=1, plot_inter=True, opt_verbose=True, r_des=1.5):
        """
        Arguments:
            sats: list of Satellite objects
            thrust: 3-element array of thrust in x, y, z direction
            objective: desired final arrangement of satellites?
            tf_horizon: horizon over which to optimize, in tf
            tf_interval: horizon over which the control inputs will be used, in tf
            plot_inter: if True, plots the intermediate optimizer results and nonlinear runs
        """
        super().__init__(sats)
        self.u = np.zeros((3, 1))
        self.horizon = tf_horizon
        self.interval = tf_interval
        self.base_res = base_res  # Used for optimization and reference trajectory generation
        self.sat = self.sats[0]
        # TODO(rgg) figure out how this works with multiple satellites, multiple segment runs
        self.scale = SatelliteScale(sat=self.sat)
        self.r_des = r_des # Final desired radius
        self.SCPn_iterations = 8
        self.plot_intermediate = plot_inter
        self.opt_verbose = opt_verbose

    def update(self, seg_num = 0):
        """
        Uses the current state of the satellites to calculate a sequence of control inputs over the horizon
        """
        const = self.scale.get_normalized_constants()
        # TODO(rgg): make this work with N satellites (not that bad?)
        # TODO(rgg): multiple SCP iterations?
        # Generate reference trajectory
        T_tan_mag = 0.5  # Tangential thrust magnitude
        c = ConstantTangentialThrustController([self.sat], T_tan_mag)
        x, t = self.run_nonlinear(c, self.horizon)
        tf_u = self.horizon
        d = Discretizer(const, use_scipy_ZOH=False, include_drag=False, include_J2=False)
        u_bar = Discretizer.extract_uk(x, t, c) # Guess inputs
        K = x.shape[1]
        nu_bar = np.zeros((7, K))

        for i in range(self.SCPn_iterations):
            print(f"Running SCP! Iteration {i}")
            K = x.shape[1] #K = int(base_res*tf)
            # Create discretizer object with default arguments (no drag, no J2)
            f = simulator.Simulator.satellite_dynamics
            # Set up optimizer and run
            print(f"Horizon: {self.horizon}")
            opt_options = { 'r_des': self.r_des,
                            'eps_r': 0.000001,
                            'eps_vr': 0.00001,
                            'eps_vt': 0.00001,
                            'eps_vn': 0.00001,
                            'tf_max': self.horizon,
                            'u_lim':[0, 0.5],
                            'w_tr': 100000,
                            'w_nu': 1E+2,
                          }
            opt = Optimizer([x], [u_bar], [nu_bar], tf_u, d, f, self.scale, verbose=self.opt_verbose)
            opt.solve_OPT(input_options=opt_options)
            # Extract outputs
            tf_u = opt.get_solved_tf(0)
            u_bar = opt.get_solved_u(0)
            nu_bar = opt.get_solved_nu(0)
            nu_sum = sum(sum(abs(nu_bar)))
            print(f"tf for optimizer: {tf_u}")
            print(f"Total virtual control effort: {nu_sum}")
            x = opt.get_solved_trajectory(0)
            print(f"Final mass from optimization: {x[6, -1]}")
            # Generate sequence controller from control outputs.
            # Controller works for simulations that end at the end of the horizon.
            self.sequence_controller=SequenceController(u=u_bar, tf_u=tf_u, tf_sim=self.interval)

            # Run nonlinear simulation with optimizer control outputs to generate a new reference trajectory
            # SequenceController needs a different timebase to be used over the entire horizon
            #c=SequenceController(u=u_opt, tf_u=tf_u, tf_sim=tf_u)
            #ref_x = x
            #if self.plot_intermediate:
            #    plot_orbit_3D(trajectories=[self.scale.redim_state(self.opt_trajectory)],
            #                                 references=[self.scale.redim_state(ref_x)],
            #                                 title=f"Reference and optimizer, iteration {i}")
            #x, t = self.run_nonlinear(c=c, tf=tf_u)
            #if self.plot_intermediate:
            #    plot_orbit_3D(trajectories=[self.scale.redim_state(self.opt_trajectory)],
            #                                 references=[self.scale.redim_state(x)],
            #                                 title=f"Actual nonlinear and optimizer, iteration {i}")
            #if self.plot_intermediate:
            #    np.savetxt(f"scpn_seg{seg_num}_iter{i}_ref_traj.csv", self.scale.redim_state(ref_x).T,delimiter=",")
            #    np.savetxt(f"scpn_seg{seg_num}_iter{i}_uopt.csv", self.scale.redim_thrust(u_opt).T,delimiter=",")
            #    np.savetxt(f"scpn_seg{seg_num}_iter{i}_opt_traj.csv", self.scale.redim_state(self.opt_trajectory).T,delimiter=",")
            #    np.savetxt(f"scpn_seg{seg_num}_iter{i}_act_traj.csv", self.scale.redim_state(x).T,delimiter=",")

        # Update horzion; THIS DEPENDS ON update() GETTING CALLED ONLY ONCE PER SIM SEGMENT
        #print(f"Horizon: {self.horizon}\tInterval:{self.interval}")
        #if self.horizon - self.interval > 0.1:
        #    self.horizon -= self.interval

    def run_nonlinear(self, c, tf):
        s = simulator.Simulator(sats=[self.sat], controller=c, scale=self.scale,
                            base_res=self.base_res, include_drag=False, include_J2=False)
        s.run(tf=tf)
        x = s.sim_data[self.sat.id] # Guess trajectory from simulation
        t = s.sim_time[self.sat.id]
        return x, t

    def get_u_func(self):
        return self.sequence_controller.get_u_func()
