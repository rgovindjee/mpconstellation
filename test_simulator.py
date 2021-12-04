# Test for sim_plotter module
import unittest
from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
from control import *
import numpy as np
import random

class TestSimulator(unittest.TestCase):

    def test_get_trajectory_ODE(self):
        # Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
        # Initial position
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        # Initial velocity
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        # Initial Thrust
        T0 = np.array([0, 0, 0])  # N
        # Initial Mass
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0, T0)
        sim = Simulator(sats=[sat])  # Use default controller
        sim.run(tf=5)  # Run for 5 orbits
        # Expect: a 3D view of the orbit
        plot_orbit_3D(sim.sim_data[sat.id].T)
        # Test saving to CSV
        sim.save_to_csv(suffix="_test")

    def test_multiple_sats(self):
        """
        Test simulation flow with 3 satellites
        """
        # Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
        # Initial position
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        # Initial velocity
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        # Initial Thrust
        T0 = np.array([0, 0, 0])  # N
        # Initial Mass
        m0 = 12200  # kg
        sats = [Satellite(r0, v0*(1+0.1*random.random()), m0, T0) for x in range(3)]
        sim = Simulator(sats=sats)  # Use default controller
        sim.run(tf=5)  # Run for 5 orbits
        # Expect: a 3D view of one orbit
        # TODO(rgg): plot all orbits
        plot_orbit_3D(sim.sim_data[sats[0].id].T)
        # Test saving to CSV
        sim.save_to_csv()

    def test_constant_thrust_controller(self):
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        T0 = np.array([0, 0, 0])  # N
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0, T0)
        # Create constant thrust controller
        # TODO(jx): fix this when the control input scaling is fixed
        c = ConstantThrustController(thrust=np.array([0., 0., 0.5]))
        sim = Simulator(sats=[sat], controller=c)
        sim.run(tf=15)  # Run for 15 orbits
        # Expect: a 3D view of the orbit
        plot_orbit_3D(sim.sim_data[sat.id].T)
        # Test saving to CSV
        sim.save_to_csv()

    def test_get_trajectory_euler(self):
        """
        Test forward Euler simulation (somewhat deprecated)
        """
        # Time
        ts = .01  # Time step in [s]
        tf_hours = 3.25  # Final time in [hr]
        tf = tf_hours * 3600  # s

        # Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
        # Initial position
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        # Initial velocity
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        # Initial Thrust
        T0 = np.array([0, 0, 0])  # N
        # Initial Mass
        m0 = 12200  # kg
        sim = Simulator()
        sim.get_trajectory(Satellite(r0, v0, m0, T0), 10, tf)
        sim.get_trajectory(Satellite(r0, v0, m0, T0), 1, tf)
        position_arr = sim.get_trajectory(Satellite(r0, v0, m0, T0), .1, tf)
        plot_orbit_2D(position_arr)
        plot_orbit_3D(position_arr)

if __name__ == '__main__':
    unittest.main()
