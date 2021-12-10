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

class TestDiscretizer(unittest.TestCase):

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


    def test_discretizer(self, use_scipy_ZOH = False, show_output=False):
        # Initial Thrust
        T_init = np.array([0.44, 0.7, 1.0])  # unitless, normalized

        #sim = Simulator(sats=[self.sat], controller = ConstantThrustController([self.sat], T_init))
        #sim.run(tf=1)  # Run for 1 orbit
        # Set up const dict
        # Designer units (pg. 20)
        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH = use_scipy_ZOH)
        # Set up inputs
        x = self.sat.get_state_vector()
        x = np.column_stack([x, x])
        u = T_init
        u = np.column_stack([u, u])
        tf = 1
        K = 2
        f = Simulator.satellite_dynamics
        A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)
        if show_output:
            print(f"A_k = {A_k}")
            print(f"B_kp = {B_kp}")
            print(f"B_kn = {B_kn}")
            print(f"Sigma_k = {Sigma_k}")
            print(f"xi_k = {xi_k}")


    def test_linearize_single(self, use_scipy_ZOH = False):
        # Initial Thrust
        T_init = np.array([0.44, 0.7, 1.0])  # unitless, normalized

        sim = Simulator(sats=[self.sat], controller = ConstantThrustController([self.sat], T_init), scale=self.scale, include_J2=False, include_drag=False)
        sim.run(tf=1)  # Run for 1 orbit

        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH = use_scipy_ZOH)
        # Set up inputs
        x = self.scale.normalize_state(self.sat.get_state_vector())
        x = np.column_stack([x, x, x])
        u = T_init
        u = np.column_stack([u, u, u])
        tf = 0.1
        K = 3
        f = Simulator.satellite_dynamics
        A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)
        # Perform the forward simulation
        x_k = x[:,0]
        x_discrete = [x_k]
        for k in range(5):
            x_k1 = A_k[0] @ x_k + B_kn[0] @ T_init + B_kp[0] @ T_init + Sigma_k[0]*tf + xi_k[0]
            x_discrete.append(x_k1)
            x_k = x_k1
        # Construct numpy array from list of rows and re-dimensionalize
        x_discrete_array = [self.scale.redim_state(np.column_stack(x_discrete))]
        # Expect: a 3D view of the orbit
        plot_orbit_3D(trajectories=x_discrete_array, references=[self.scale.redim_state(sim.sim_data[self.sat.id])], use_mayavi = True)


    def test_linearize_many(self):
        # Initial Thrust
        T_init = np.array([0.44, 0.7, 1.0])  # unitless, normalized
        # Adjustable parameters
        tf = 1
        base_res = 100
        # Use simulator to generate a reference (guess) trajectory
        sim = Simulator(sats=[self.sat], controller=ConstantThrustController([self.sat], T_init), scale=self.scale, base_res=base_res, include_drag = False, include_J2 = False)
        sim.run(tf=tf)
        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH=False, include_drag = False, include_J2 = False)
        # Set up inputs
        x = sim.sim_data[self.sat.id] # Guess trajectory from simulation
        K = x.shape[1] #K = int(base_res*tf)
        print(f"final mass: {x[-1, -1]}")
        u = np.tile(T_init, (3, K)) # Inputs constant for this example
        f = Simulator.satellite_dynamics
        A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)
        # Perform the forward simulation
        x_k = x[:,0]
        x_discrete = [x_k]
        print("K: {K}")
        for k in range(K-1):
            x_k1 = A_k[k] @ x_k + B_kn[k] @ u[:,k] + B_kp[k] @ u[:,k+1] + Sigma_k[k]*tf + xi_k[k]
            x_discrete.append(x_k1)
            x_k = x_k1
        # Construct numpy array from list of rows and re-dimensionalize
        x_discrete_array = [self.scale.redim_state(np.column_stack(x_discrete))]
        # Expect: a 3D view of the orbit
        plot_orbit_3D(trajectories=x_discrete_array, references=[self.scale.redim_state(x)], use_mayavi=True)


    def test_linearize_tangential(self):
        # Initial Thrust
        T_tan_mag = 0.5  # Tangential Trhust magnitude
        c = ConstantTangentialThrustController([self.sat], T_tan_mag)
        # Adjustable parameters
        tf = 2
        base_res = 100
        # Use simulator to generate a reference (guess) trajectory
        sim = Simulator(sats=[self.sat], controller=c, scale=self.scale, base_res=base_res, include_drag = False, include_J2 = False)
        sim.run(tf=tf)
        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH=False, include_drag = False, include_J2 = False)
        # Set up inputs
        x = sim.sim_data[self.sat.id] # Guess trajectory from simulation
        K = x.shape[1] #K = int(base_res*tf)
        print(f"final mass: {x[-1, -1]}")
        u = d.extract_uk(x, sim.sim_time[self.sat.id], c) # Guess inputs
        f = Simulator.satellite_dynamics
        A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)
        # Perform the forward simulation
        x_k = x[:,0] # Start with initial conditions
        x_discrete = [x_k]
        print("K: {K}")
        for k in range(K-1):
            x_k1 = A_k[k] @ x_k + B_kn[k] @ u[:,k] + B_kp[k] @ u[:,k+1] + Sigma_k[k]*tf + xi_k[k]
            x_discrete.append(x_k1)
            x_k = x_k1
        # Construct numpy array from list of rows and re-dimensionalize
        x_discrete_array = [self.scale.redim_state(np.column_stack(x_discrete))]
        # Expect: a 3D view of the orbit
        plot_orbit_3D(trajectories=x_discrete_array, references=[self.scale.redim_state(x)], use_mayavi=True)

    def test_custom_ZOH(self):
        print("Results with custom ZOH")
        self.test_single_state(False)
        print("\n")
        print("Results with built-in ZOH")
        self.test_single_state(True)

if __name__ == '__main__':
    unittest.main()
