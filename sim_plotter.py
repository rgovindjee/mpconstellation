# A collection of methods to plot simulator output
import matplotlib.pyplot as plt
import numpy as np
import sys
from constants import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


# Import mayavi if installed
try:
    from mayavi import mlab
    from tvtk.api import tvtk
    _mayavi_available = True
except:
    _mayavi_available = False

# TODO: also plot on a useful projection, not just a top-down view
def plot_orbit_2D(trajectories):
    """
    Arguments:
        trajectories: N list of 3 x T arrays of x, y, z positions, or 7 x T
                      arrays of state vectors.
    Plots trajectory on a 2D representation of the Earth
    """
    fig, ax = plt.subplots()
    for t in trajectories:
        ax.plot(t[0, :], t[1, :])
    earth = plt.Circle((0, 0), R_EARTH, color='b')
    ax.add_patch(earth)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_orbit_3D(trajectories, references = [], use_mayavi = True, show_quiver=True):
    """
    Arguments:
        trajectories: N list of 3 x T arrays of x, y, z positions, or 7 x T
                      arrays of state vectors.
        references: K list of 3 x T arrays of x, y, z positions, or 7 x T
                    arrays of state vectors. Reference trajectories colored green.
        use_mayavi: Boolean, use mayavi or matplotlib.
    Plots trajectories on a 3D representation of the Earth
    """
    if _mayavi_available and use_mayavi:
        # Generate an Earth
        fig = mlab.figure(size=(600,600))
        img = tvtk.JPEGReader()
        img.file_name = "blue_marble.jpg"
        texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
        sphere = tvtk.TexturedSphereSource(radius=R_EARTH, theta_resolution=180, phi_resolution=180)
        sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
        sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
        fig.scene.add_actor(sphere_actor)
        #earth = mlab.points3d(0, 0, 0, 2*R_EARTH,scale_factor = 1, resolution = 1024, opacity=0.8)
        for r in references:
            mlab.plot3d(r[0,:], r[1,:], r[2,:], tube_radius = 50000, color=(0,1,0))
        # Plot trajectories
        for t in trajectories:
            if show_quiver and t.shape[0] == 7:
                idx = np.linspace(0, t.shape[1]-1, 40, dtype=int)
                mlab.quiver3d(t[0,idx], t[1,idx], t[2,idx], t[3,idx], t[4,idx], t[5,idx],mode='cone', scale_factor=100, color=(0.5,0,0))
            mlab.plot3d(t[0,:], t[1,:], t[2,:], tube_radius=50000, color=(1,0,0))
        mlab.show()
    else:
        ax = plt.axes(projection='3d')
        # Plot Earth Sphere
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        xm = R_EARTH * np.outer(np.cos(phi), np.sin(theta))
        ym = R_EARTH * np.outer(np.sin(phi), np.sin(theta))
        zm = R_EARTH * np.outer(np.ones(np.size(phi)), np.cos(theta))
        ax.plot_surface(xm, ym, zm)
        # Plot reference trajectories
        for r in references:
            ax.plot3D(r[0,:], r[1,:], r[2,:], color=(1,0,0))
        # Plot remaining trajectories
        for t in trajectories:
            ax.plot3D(t[0, :], t[1, :], t[2, :])
        #x.set_box_aspect(aspect=(np.ptp(xm), np.ptp(ym), np.ptp(zm)))
        plt.show()
