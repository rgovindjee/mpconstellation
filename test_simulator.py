# Test for sim_plotter module
import unittest
from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
import numpy as np

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
        sim = Simulator()
        pos = sim.get_trajectory_ODE(sat, 10) 
        # Expect: a 3D view of the orbit
        plot_orbit_3D(pos.T)
        # Test saving to CSV
        sim.save_to_csv()

    def test_get_trajectory_euler(self):
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