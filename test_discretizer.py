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

    def test_single_state(self):
        # Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
        # Initial position
        r_init = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        # Initial velocity
        v_init = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        # Initial Thrust
        T_init = np.array([0.44, 0.7, 1.0])  # unitless, normalized
        # Initial Mass
        m_init = 12200  # kg
        sat = Satellite(r_init, v_init, m_init, T_init)
        sim = Simulator(sats=[sat], controller = ConstantThrustController([sat], T_init))  # Use default controller
        sim.run(tf=1)  # Run for 1 orbit

        # Set up const dict
        # Designer units (pg. 20)
        r0  = np.linalg.norm(sat.position)
        s0 = 2*np.pi*np.sqrt(r0**3/MU_EARTH)
        v0 = r0/s0
        a0 = r0/s0**2
        m0 = sat.mass
        T0 = m0*r0/s0**2
        mu0 = r0**3/s0**2
        # Normalized state vector (pg. 21)
        y0 = np.concatenate([sat.position/r0, sat.velocity/v0, np.array([sat.mass/m0])])
        # Normalize system parameters (pg. 21)
        const = constants.Constants(MU=MU_EARTH/mu0, R_E=R_EARTH/r0, J2=J2, G0=G0/a0, ISP=ISP/s0, S=S/r0**2, R0=r0, RHO=m0/r0**3)
        print(f"consts: {const.G0}, {const.ISP}")
        # Create discretizer object with default arguments (no drag, no J2)
        d = Discretizer(const)
        # Set up inputs
        x = np.hstack([sat.position/r0, sat.velocity/v0, [sat.mass/m0]])
        x = np.column_stack([x, x])
        u = T_init
        u = np.column_stack([u, u])
        tf = 1
        K = 2
        f = Simulator.satellite_dynamics
        A_k, B_kp, B_kn, Sigma_k, xi_k = d.discretize(f, x, u, tf, K)
        print(f"A_k = {A_k}")
        print(f"B_kp = {B_kp}")
        print(f"B_kn = {B_kn}")
        print(f"Sigma_k = {Sigma_k}")
        print(f"xi_k = {xi_k}")
        # Expect: a 3D view of the orbit
        #plot_orbit_3D(sim.sim_data[sat.id].T)

if __name__ == '__main__':
    unittest.main()
