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
        self.r_init = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        # Initial velocity
        self.v_init = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        # Initial Mass
        self.m_init = 12200  # kg
        # Create satellite object
        self.sat = Satellite(self.r_init, self.v_init, self.m_init)
        # Create a scaling object
        self.scale = SatelliteScale(sat=self.sat)
        # Normalize system parameters (pg. 21)
        self.const = self.scale.get_normalized_constants()

    def test_optimizer_single(self):
        # Initial Thrust
        T_tan_mag = 0.0005  # Tangential thrust magnitude
        c = ConstantTangentialThrustController([self.sat], T_tan_mag)
        #c = Controller()
        # Adjustable parameters
        tf = 2
        base_res = 100
        # Use simulator to generate a reference (guess) trajectory
        sim = Simulator(sats=[self.sat], controller=c, scale=self.scale, base_res=base_res, include_drag = False, include_J2 = False)
        sim.run(tf=tf)
        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH=False, include_drag=False, include_J2=False)
        # Set up inputs
        x = sim.sim_data[self.sat.id] # Guess trajectory from simulation
        K = x.shape[1] #K = int(base_res*tf)
        print(f"final mass: {x[-1, -1]}")
        u_bar = Discretizer.extract_uk(x, sim.sim_time[self.sat.id], c) # Guess inputs
        nu_bar = np.zeros((7, K))
        f = Simulator.satellite_dynamics
        # Create Optimizer object
        #opt_options = {'r_des':np.linalg.norm(x[0:3, -1])}
        opt_options = {'r_des':1.4879}

        opt = Optimizer([x], [u_bar], [nu_bar], tf, d, f, self.scale)
        opt.get_constraint_terms()
        opt.solve_OPT(input_options=opt_options)
        print(f"model:\n {opt.model}")
        # Expect: a 3D view of the orbit
        # Get 0th satellite as there is only one
        print(f"Last opt x:\n{opt.get_solved_trajectory(0)[:,-5:]}")
        opt_trajectory = self.scale.redim_state(opt.get_solved_trajectory(0))


        plot_orbit_3D(trajectories=[opt_trajectory],
                       references=[self.scale.redim_state(x)],
                       use_mayavi=True)

        # Simulate nonlinear trajectory for more orbits using control outputs
        tf_sim = 5
        tf_u = opt.get_solved_tf(0)
        print(f"tf_u: {tf_u}")
        u_opt = opt.get_solved_u(0)
        opt.plot_normalized_thrust(u_opt)
        c_opt = SequenceController(u=u_opt, tf_u=tf_u, tf_sim=tf_sim)
        sim = Simulator(sats=[self.sat], controller=c_opt, scale=self.scale, base_res=base_res, include_drag = False, include_J2 = False)
        sim.run(tf=tf_sim)
        x_forward = sim.sim_data[self.sat.id] # Guess trajectory from simulation
        plot_orbit_3D(trajectories=[opt_trajectory],
                      references=[self.scale.redim_state(x_forward)],
                      use_mayavi=True)

if __name__ == '__main__':
    unittest.main()
