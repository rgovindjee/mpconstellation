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
        #r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        #v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        r0 = np.array([6.920659696230498E+03, 0, 0]) * 1000 # m
        v0 = np.array([0, 7.588871358113800, 0]) * 1000 # m/s
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        sats = [sat]
        res = 100
        tf = 2
        num_segments = 1
        tf_interval = tf / num_segments
        # Create the controller
        c = OptimalController(  sats=sats, base_res=30, tf_horizon=tf,
                                tf_interval=tf_interval, plot_inter=True, opt_verbose=False)
        scale = SatelliteScale(sat=sat)
        sim = Simulator(sats=sats, controller=c, scale=scale, base_res=res, verbose=True)
        sim.run_segments(tf=tf, num_segments=num_segments)
        # Calculate circular speed and verify with actual tangential speed
        const = scale.get_normalized_constants()
        x_opt = c.opt_trajectory
        x_act = sim.sim_data[sat.id]
        alt_final_c = np.linalg.norm(x_opt[0:3,-1])
        alt_final_act = np.linalg.norm(x_act[0:3,-1])
        print(f"Expected final altitude: 2; Controller: {alt_final_c}; Actual: {alt_final_act}")
        Vc_final_c = np.sqrt(const.MU/alt_final_c)
        Vc_final_act = np.sqrt(const.MU/alt_final_act)
        # Get tangential, normal, radial vectors:
        r_f = x_opt[0:3,-1]
        r_hat_f = r_f/np.linalg.norm(r_f)
        v_f = x_opt[3:6,-1]
        h = np.cross(r_f, v_f)
        h_hat = h/np.linalg.norm(h)
        t_hat = np.cross(h_hat, r_hat_f)
        Vr = np.dot(v_f, r_hat_f)
        Vt = np.dot(v_f, t_hat)
        Vn = np.dot(v_f, h_hat)

        print(f"Expected controller circular speed (Vt):\n{Vc_final_c}\nController actual velocity:\nVr:{Vr} Vt:{Vt} Vn:{Vn}\n")

        # Get tangential, normal, radial vectors:
        r_f_act = x_act[0:3,-1]
        r_hat_f_act = r_f_act/np.linalg.norm(r_f_act)
        v_f_act = x_act[3:6,-1]
        h_act= np.cross(r_f_act, v_f_act)
        h_hat_act = h_act/np.linalg.norm(h_act)
        t_hat_act = np.cross(h_hat_act, r_hat_f_act)
        Vr_act = np.dot(v_f_act, r_hat_f_act)
        Vt_act = np.dot(v_f_act, t_hat_act)
        Vn_act = np.dot(v_f_act, h_hat_act)
        print(f"Expected actual circular speed (Vt):\n{Vc_final_act}\nActual velocity:\nVr:{Vr_act} Vt:{Vt_act} Vn:{Vn_act}\n")

        # Propogate, check for circularity
        print(f"Final satellite mass: {sat.mass/m0}")
        sim2 = Simulator(sats=[sat], scale=scale, base_res=res, verbose=False)
        sim2.run(tf=5)
        x_sim_ff1 = sim2.sim_data[sat.id]
        radius = np.linalg.norm(x_sim_ff1[0:3,:], axis=0)
        plot2D(radius, title='Propogated satellite 1')

        # Propogate a "control" satellite
        sat2 = Satellite(position=np.array([alt_final_c*scale._r0, 0, 0]), velocity=np.array([0, Vt*scale._v0, 0]), mass=sat.mass)
        sim3 = Simulator(sats=[sat2], scale=scale, base_res=res, verbose=False)
        sim3.run(tf=5)
        x_sim_ff2 = sim3.sim_data[sat2.id]
        radius = np.linalg.norm(x_sim_ff2[0:3,:], axis=0)
        plot2D(radius, title='Propogated satellite 2 (benchmark)')

        # Expect: a 3D view of orbits for all sats
        np.savetxt("mpc_act_traj.csv", scale.redim_state(x_act).T, delimiter=",")
        np.savetxt("mpc_con_traj.csv", scale.redim_state(x_opt).T, delimiter=",")
        np.savetxt("mpc_fwd_traj.csv", scale.redim_state(x_sim_ff1).T, delimiter=",")
        plot_orbit_3D(trajectories=[scale.redim_state(sim.sim_data[sats[i].id]) for i in range(len(sats))] + [scale.redim_state(x_sim_ff1)],
         references=[scale.redim_state(c.opt_trajectory)])

    def test_run_segments(self):
        print("Testing multi-segment wrapper function with two satellites")
        r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
        v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
        m0 = 12200  # kg
        sat = Satellite(r0, v0, m0)
        sat2 = Satellite(r0, v0*1.1, m0)
        sats = [sat, sat2]
        res = 100
        c = ConstantTangentialThrustController(sats=sats, tangential_thrust=0.5)
        # Create satellite scale object
        scale = SatelliteScale(sat=sat)
        sim = Simulator(sats=sats, scale=scale, base_res=res, controller=c)
        tf = 3
        sim.run_segments(tf=tf, num_segments=4)
        print(f"Expected time shape: ({tf*res},)")
        print(f"Got: {sim.sim_time[sat.id].shape}")

        final_alt = np.linalg.norm(sim.sim_data[sat.id][0:3, -1])
        final_mass = np.linalg.norm(sim.sim_data[sat.id][6, -1])
        print(f"Final altitude: {final_alt}")
        print(f"Final mass: {final_mass}")

        ## Expect: a 3D view of orbits for all sats
        #plot_orbit_3D([scale.redim_state(sim.sim_data[sats[i].id]) for i in range(len(sats))])

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
