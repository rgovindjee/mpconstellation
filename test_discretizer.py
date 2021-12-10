import unittest
from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
from linearize_discretize import Discretizer
from control import *
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
        # Initial Thrust
        self.T_init = np.array([0.44, 0.7, 1.0])  # unitless, normalized
        # Initial Mass
        self.m_init = 12200  # kg
        self.sat = Satellite(self.r_init, self.v_init, self.m_init, self.T_init)
        self.r0  = np.linalg.norm(self.sat.position)
        self.s0 = 2*np.pi*np.sqrt(self.r0**3/MU_EARTH)
        self.v0 = self.r0/self.s0
        self.a0 = self.r0/self.s0**2
        self.m0 = self.sat.mass
        self.T0 = self.m0*self.r0/self.s0**2
        self.mu0 = self.r0**3/self.s0**2
        # Normalize system parameters (pg. 21)
        self.const = constants.Constants(MU=MU_EARTH/self.mu0, R_E=R_EARTH/self.r0, J2=J2, G0=G0/self.a0, ISP=ISP/self.s0, S=S/self.r0**2, R0=self.r0, RHO=self.m0/self.r0**3)

    def test_single_state(self, use_scipy_ZOH = False, show_output=False):
        sim = Simulator(sats=[self.sat], controller = ConstantThrustController([self.sat], self.T_init))
        sim.run(tf=1)  # Run for 1 orbit
        # Set up const dict
        # Designer units (pg. 20)
        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH = use_scipy_ZOH)
        # Set up inputs
        x = np.hstack([self.sat.position/self.r0, self.sat.velocity/self.v0, [self.sat.mass/self.m0]])
        x = np.column_stack([x, x])
        u = self.T_init
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

    def test_visualize_linear(self, use_scipy_ZOH = False):
        sim = Simulator(sats=[self.sat], controller = ConstantThrustController([self.sat], self.T_init))
        sim.run(tf=1)  # Run for 1 orbit
        # Show expected results
        plot_orbit_3D(sim.sim_data[self.sat.id].T)

        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH = use_scipy_ZOH)
        # Set up inputs
        x = np.hstack([self.sat.position/self.r0, self.sat.velocity/self.v0, [self.sat.mass/self.m0]])
        x = np.column_stack([x, x])
        u = self.T_init
        u = np.column_stack([u, u])
        tf = 0.1
        K = 2
        f = Simulator.satellite_dynamics
        A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)
        # Perform the forward simulation
        x_k = x[:,0]
        pos_discrete = [x_k[0:3]]
        for k in range(5):
            x_k1 = A_k[0] @ x_k + B_kn[0] @ self.T_init + B_kp[0] @ self.T_init + Sigma_k[0]*tf + xi_k[0]
            pos_discrete.append(x_k1[0:3])
            x_k = x_k1
        # Construct numpy array from list of rows and re-dimensionalize
        pos_discrete_array = np.array(pos_discrete)*self.r0
        # Expect: a 3D view of the orbit
        plot_orbit_3D(pos_discrete_array)

    def test_linearize_many(self):
        tf = 3.
        base_res = 100
        K = int(base_res*tf)
        sim = Simulator(sats=[self.sat], controller=ConstantThrustController([self.sat], self.T_init), base_res=base_res)
        sim.run(tf=tf)  # Run for 1 orbit
        # Show expected results
        plot_orbit_3D(sim.sim_data[self.sat.id].T)

        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(self.const, use_scipy_ZOH=False)
        # Set up inputs
        x = sim.sim_full_state[self.sat.id]  # From simulation
        u = np.tile(self.T_init, (3, K))
        #u = np.zeros((3, K))
        f = Simulator.satellite_dynamics
        A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)
        # Perform the forward simulation
        x_k = x[:,0]
        pos_discrete = [x_k[0:3]]
        for k in range(K-1):
            x_k1 = A_k[k] @ x_k + B_kn[k] @ u[:,k] + B_kp[k] @ u[:,k] + Sigma_k[k]*tf + xi_k[k]
            pos_discrete.append(x_k1[0:3])
            x_k = x_k1
        print(f"final mass: {x[-1, -1]}")
        # Construct numpy array from list of rows and re-dimensionalize
        pos_discrete_array = np.array(pos_discrete)*self.r0
        # Expect: a 3D view of the orbit
        plot_orbit_3D(pos_discrete_array)

    def test_custom_ZOH(self):
        print("Results with custom ZOH")
        self.test_single_state(False)
        print("\n")
        print("Results with built-in ZOH")
        self.test_single_state(True)

if __name__ == '__main__':
    unittest.main()
