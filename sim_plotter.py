# A collection of methods to plot simulator output
import matplotlib.pyplot as plt
import numpy as np
from constants import *

# TODO: also plot on a useful projection, not just a top-down view
def plot_orbit_2D(position_arr):
    """
    Arguments:
        position_arr: Tx3 array of x, y, z coordinates for a single satellite
    Plots trajectory on a 2D representation of the Earth
    """
    fig, ax = plt.subplots()
    ax.plot(position_arr[:, 0], position_arr[:, 1])
    earth = plt.Circle((0, 0), R_EARTH, color='g')
    ax.add_patch(earth)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_orbit_3D(position_arr):
    """
    Arguments:
        position_arr: Tx3 array of x, y, z coordinates for a single satellite
    Plots trajectory on a 3D representation of the Earth
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth Sphere
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    xm = R_EARTH * np.outer(np.cos(phi), np.sin(theta))
    ym = R_EARTH * np.outer(np.sin(phi), np.sin(theta))
    zm = R_EARTH * np.outer(np.ones(np.size(phi)), np.cos(theta))
    ax.plot_surface(xm, ym, zm)

    # Plot satellite trajectory
    ax.plot(position_arr[:, 0], position_arr[:, 1], position_arr[:, 2])
    plt.show()