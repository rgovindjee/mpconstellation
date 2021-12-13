# Test for sim_plotter module
import unittest
from sim_plotter import *
from simulator import Simulator
from satellite import Satellite
from satellite_scale import SatelliteScale
from control import *
import numpy as np
import logging
import random
# TODO: get rid of annoying warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestSimulator(unittest.TestCase):

    def test_get_trajectory_ODE(self):
        # Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
        # Initial position
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        # Initial velocity
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
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

    def test_run_segment(self):
        print("Testing multi-segment sim run with one satellite")
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        sats = [sat]
        res = 20
        # Create satellite scale object
        scale = SatelliteScale(sat=sat)
        sim = Simulator(sats=sats, scale=scale, base_res=res)  # Use default controller
        print("Testing first orbit")
        sim.run_segment(tf=1)  # Run for 1 orbit
        print("Testing second orbit")
        sim.run_segment(tf=1)  # Run for 1 orbit
        print("Testing third and fourth orbits")
        sim.run_segment(tf=2)  # Run for 2 orbits
        print(f"Expected time shape: ({4*res},)")
        print(f"Got: {sim.sim_time[sat.id].shape}")
        # Expect: a 3D view of orbits for all sats
        #plot_orbit_3D([scale.redim_state(sim.sim_data[sats[i].id]) for i in range(len(sats))])

    def test_mpc(self):
        print("Testing MPC controller with one satellite")
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        sats = [sat]
        res = 100
        tf = 2
        num_segments = 1
        tf_interval = tf / num_segments
        # Create the controller
        c = OptimalController(sats=sats, base_res=50, tf_horizon=tf, tf_interval=tf_interval)
        scale = SatelliteScale(sat=sat)
        sim = Simulator(sats=sats, controller=c, scale=scale, base_res=res, verbose=True)
        sim.run_segments(tf=tf, num_segments=num_segments)
        # Calculate circular speed and verify with actual tangential speed
        const = scale.get_normalized_constants()
        x_opt = c.opt_trajectory
        alt_final = np.linalg.norm(x_opt[0:3,-1])
        print(f"Expected final altitide: 1.5; Actual: {alt_final}")
        Vc_final = np.sqrt(const.MU/alt_final)
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
        print(f"Expected circular speed (Vt):\n{Vc_final}\nActual Velocity:\nVr:{Vr} Vt:{Vt} Vn:{Vn}\n")

        # Expect: a 3D view of orbits for all sats
        plot_orbit_3D(trajectories=[scale.redim_state(sim.sim_data[sats[i].id]) for i in range(len(sats))],
                      references=[scale.redim_state(c.opt_trajectory)])

    def test_run_segments(self):
        print("Testing multi-segment wrapper function with two satellites")
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        sat2 = Satellite(r0, v0*1.1, m0)
        sats = [sat, sat2]
        res = 20
        # Create satellite scale object
        scale = SatelliteScale(sat=sat)
        sim = Simulator(sats=sats, scale=scale, base_res=res)  # Use default controller
        tf = 4
        sim.run_segments(tf=tf, num_segments=8)
        print(f"Expected time shape: ({tf*res},)")
        print(f"Got: {sim.sim_time[sat.id].shape}")

        # Expect: a 3D view of orbits for all sats
        plot_orbit_3D([scale.redim_state(sim.sim_data[sats[i].id]) for i in range(len(sats))])

    def test_constant_thrust_controller(self):
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        scale = SatelliteScale(sat=sat)
        # Create constant thrust controller
        c = ConstantThrustController(thrust=np.array([0., 0., 0.1]))
        sim = Simulator(sats=[sat], controller=c, scale=scale)
        sim.run(tf=15)  # Run for 15 orbits
        # Expect: a 3D view of the orbit
        plot_orbit_3D([scale.redim_state(sim.sim_data[sat.id])])
        # Test saving to CSV
        sim.save_to_csv()

    def test_tangential_thrust_controller(self):
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        scale = SatelliteScale(sat = sat)
        # Create constant thrust controller
        c = ConstantTangentialThrustController(tangential_thrust=0.1)
        sim = Simulator(sats=[sat], controller=c, scale = scale)
        sim.run(tf=5)  # Run for 15 orbits
        # Expect: a 3D view of the orbit
        plot_orbit_3D([scale.redim_state(sim.sim_data[sat.id])])
        # Test saving to CSV
        sim.save_to_csv()

if __name__ == '__main__':
    unittest.main()
