import unittest
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

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        # Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
        # Initial position
        self.r_init = np.array([(R_EARTH+250000), 0.0, 0.0])  # m
        # Initial velocity
        Vc = np.sqrt(MU_EARTH/np.linalg.norm(self.r_init))
        self.v_init = np.array([0, Vc, 0]) # m/s
        # Initial Mass
        self.m_init = 100 # kg
        # Create satellite object
        self.sat = Satellite(self.r_init, self.v_init, self.m_init)
        # Create a scaling object
        self.scale = SatelliteScale(sat=self.sat)
        # Normalize system parameters (pg. 21)
        self.const = self.scale.get_normalized_constants()

    def test_optimizer_single(self):
        print(self.r_init)
        print(self.scale.normalize_state(np.concatenate([self.r_init, self.v_init, np.array([self.m_init])])))
        # Initial Thrust
        T_tan_mag = 0.1/self.scale._T0 # Max tangential thrust is 0.1N, normalized
        c = ConstantTangentialThrustController([self.sat], T_tan_mag)
        # Adjustable parameters
        tf = 26.3769
        base_res = 100
        # Use simulator to generate a reference (guess) trajectory
        sim = Simulator(sats=[self.sat], controller=c, scale=self.scale, base_res=base_res, include_drag = False, include_J2 = False)
        sim.run(tf=tf)
        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH=False, include_drag=False, include_J2=False)
        # Set up inputs
        x_bar = sim.sim_data[self.sat.id] # Guess trajectory from simulation
        K = x_bar.shape[1] #K = int(base_res*tf)
        print(f"final mass: {x_bar[-1, -1]}")
        print(f"final alt: {(np.linalg.norm(self.scale.redim_state(x_bar)[0:3,-1]) - R_EARTH)/1000}")
        u_bar = Discretizer.extract_uk(x_bar, sim.sim_time[self.sat.id], c) # Guess inputs
        nu_bar = np.zeros((7, K))
        f = Simulator.satellite_dynamics
        
        # Create Optimizer object
        opt_options = { 'min_mass':80/self.scale._m0,
                        'u_lim':[0, 0.1/self.scale._T0],
                        'r_lim':[(200000+R_EARTH)/self.scale._r0, (700000+R_EARTH)/self.scale._r0],
                        'r_des':(500000+R_EARTH)/self.scale._r0,
                        'eps_r': 0.01,
                        'eps_vr': 0.0001,
                        'eps_vn': 0.0001,
                        'eps_vt': 0.0001,
                        'tf_max':50,
                        'w_nu':300,
                        'w_tr':2}

        opt = Optimizer([x_bar], [u_bar], [nu_bar], tf, d, f, self.scale)
        opt.solve_OPT(input_options=opt_options)
        # Expect: a 3D view of the orbit
        # Get 0th satellite as there is only one
        x_opt = opt.get_solved_trajectory(0)
        opt_trajectory = self.scale.redim_state(x_opt)
        plot_orbit_3D(trajectories=[opt_trajectory],
                       references=[self.scale.redim_state(x_bar)],
                       use_mayavi=True)

        # Calculate circular speed and verify with actual tangential speed
        alt_final = np.linalg.norm(x_opt[0:3,-1])
        Vc_final = np.sqrt(self.const.MU/alt_final)
        # Get tangential, normal, radial vectors:
        r_f = x_opt[0:3,-1]
        r_hat_f = r_f/np.linalg.norm(r_f)
        v_f = x_opt[3:6,-1]
        v_hat_f = v_f/np.linalg.norm(v_f)
        h = np.cross(r_f, v_f)
        h_hat = h/np.linalg.norm(h)
        t_hat = np.cross(h_hat, r_hat_f)
        Vr = np.dot(v_f, r_hat_f)
        Vt = np.dot(v_f, t_hat)
        Vn = np.dot(v_f, h_hat)
        print(f"Expected circular velocity:\n{Vc_final}\nActual Velocity:\nVr:{Vr} Vt:{Vt} Vn:{Vn}\n")

        # Simulate nonlinear trajectory for more orbits using control outputs
        tf_u = opt.get_solved_tf(0)
        tf_sim = tf_u + 10
        print(f"tf_u: {tf_u}")
        u_opt = opt.get_solved_u(0)
        opt.plot_thrust(x_opt, u_opt, self.scale)
        c_opt = SequenceController(u=u_opt, tf_u=tf_u, tf_sim=tf_sim)
        sim = Simulator(sats=[self.sat], controller=c_opt, scale=self.scale, base_res=base_res, include_drag = False, include_J2 = False)
        sim.run(tf=tf_sim)
        x_forward = sim.sim_data[self.sat.id] # Guess trajectory from simulation
        plot_orbit_3D(trajectories=[opt_trajectory],
                      references=[self.scale.redim_state(x_forward)],
                      use_mayavi=True)


if __name__ == '__main__':
    unittest.main()
