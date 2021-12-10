# Test for sim_plotter module
import unittest
from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
from satellite_scale import SatelliteScale
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
        sat = Satellite(r0, v0, m0)
        # Create simulator scale object
        scale = SatelliteScale(sat = sat)
        sim = Simulator(sats=[sat], scale=scale)  # Use default controller
        sim.run(tf=5)  # Run for 5 orbits
        # Expect: a 3D view of the orbit
        plot_orbit_3D([scale.redim_state(sim.sim_data[sat.id])])
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
        sats = [Satellite(r0, v0*(1+0.1*random.random()), m0) for x in range(3)]
        # Create satellite scale object
        scale = SatelliteScale(sat = sats[0])
        sim = Simulator(sats=sats, scale=scale)  # Use default controller
        sim.run(tf=5)  # Run for 5 orbits
        # Expect: a 3D view of orbits for all sats
        plot_orbit_3D([scale.redim_state(sim.sim_data[sats[i].id]) for i in range(len(sats))])
        # Test saving to CSV
        sim.save_to_csv()


    def test_constant_thrust_controller(self):
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        T0 = np.array([0, 0, 0])  # N
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        scale = SatelliteScale(sat = sat)
        # Create constant thrust controller
        c = ConstantThrustController(thrust=np.array([0., 0., 0.1]))
        sim = Simulator(sats=[sat], controller=c, scale = scale)
        sim.run(tf=15)  # Run for 15 orbits
        # Expect: a 3D view of the orbit
        plot_orbit_3D([scale.redim_state(sim.sim_data[sat.id])])
        # Test saving to CSV
        sim.save_to_csv()


if __name__ == '__main__':
    unittest.main()
