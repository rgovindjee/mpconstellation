# Test for sim_plotter module
import unittest
from sim_plotter import *
import numpy as np

class TestPlotter(unittest.TestCase):

    def test_plot_orbit_2D(self):
        # Generate a trajectory to plot with some eccentricity
        th = np.linspace(0, 7, 1000)
        radius = 7500000  # m
        x = radius*np.sin(th)
        y = radius*1.3*np.cos(th)
        z = (radius/10)*np.cos(th*5)
        pos = np.array([x, y, z])
        # Expect: a 2D view of the orbit from one of Earth's poles
        plot_orbit_2D([pos])

    def test_plot_orbit_3D(self):
        # Generate a trajectory to plot with some interesting oscillation in z
        th = np.linspace(0, 7, 1000)
        radius = 7000000  # m
        x = radius*np.sin(th)
        y = radius*np.cos(th)
        z = (radius/10)*np.cos(th*5)
        pos = np.array([x, y, z])
        # Expect: a 3D view of the orbit
        plot_orbit_3D(trajectories=[pos], references=[], use_mayavi=True)

if __name__ == '__main__':
    unittest.main()