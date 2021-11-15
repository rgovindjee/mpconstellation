import numpy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mu_earth = 3.986004418E5  # km3 s−2
r_earth = 6.378E3  # m
g0 = 9.80665  # m/s2
Isp = 500  # s


# position - 3 x 1
# velocity - 3 x 1
# The position and velocity vectors are coordinated in an inertial reference frame placed
# at the center of the central body with its z-axis aligned with rotation axis of the central body

# thrust - 3 x 1

class Satellite:
    def __init__(self, position, velocity, mass, thrust):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.thrust = thrust

    def __str__(self):
        return f"r: {self.position}"


def update_satellite_state(sat, time_step):
    # Update Position
    position_next = sat.velocity * time_step + sat.position

    # Update Velocity
    accel = -(mu_earth / np.linalg.norm(sat.position) ** 3) * sat.position + sat.thrust / sat.mass
    velocity_next = accel * time_step + sat.velocity

    # Update Mass
    mass_dot = np.linalg.norm(sat.thrust) / g0 * Isp
    mass_next = mass_dot * time_step + sat.mass

    # Update Satellite object
    sat.position = position_next
    sat.velocity = velocity_next
    sat.mass = mass_next


def get_trajectory(sat, ts, tf):
    n = int(tf / ts)
    position = numpy.zeros(shape=(n, 3))
    init = sat.position
    for i in range(n):
        update_satellite_state(sat, ts)
        position[i, :] = sat.position
        if np.linalg.norm(position[i, :]) < r_earth:
            # If magnitude of position vector for any point is less than the earth's
            # radius, that means the satellite has crashed into earth
            print("Crashed!")
            print(position[i, :])
    final = sat.position
    print(f'Error with ts={ts}')
    print(np.linalg.norm(final) - np.linalg.norm(init))
    return position


def plot_orbit_2D(position_arr):
    fig, ax = plt.subplots()
    ax.plot(position_arr[:, 0], position_arr[:, 1])
    earth = plt.Circle((0, 0), r_earth, color='g')
    ax.add_patch(earth)
    plt.show()


def plot_orbit_3D(position_arr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth Sphere
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    xm = r_earth * np.outer(np.cos(phi), np.sin(theta))
    ym = r_earth * np.outer(np.sin(phi), np.sin(theta))
    zm = r_earth * np.outer(np.ones(np.size(phi)), np.cos(theta))
    ax.plot_surface(xm, ym, zm)

    # Plot satellite trajectory
    ax.plot(position_arr[:, 0], position_arr[:, 1], position_arr[:, 2])
    plt.show()


# Not Used
# altitude_km = 5  # km
# height = altitude_km + r_earth  # km

# Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
# Initial position
r0 = np.array([5371.4806, -4133.1393, 1399.9594])  # km
# Initial velocity
v0 = np.array([4.6921, 4.9848, -3.2752])  # km/s
# Initial Thrust
T0 = np.array([0, 0, 0])  # N
# Initial Mass
m0 = 12200  # kg
# Time
ts = .01  # Time step in [s]
tf_hours = 3.25  # Final time in [hr]
tf = tf_hours * 3600  # s

# get_trajectory(Satellite(r0, v0, m0, T0), 10, tf)
# get_trajectory(Satellite(r0, v0, m0, T0), 1, tf)
position_arr = get_trajectory(Satellite(r0, v0, m0, T0), .1, tf)
# get_trajectory(Satellite(r0, v0, m0, T0), .01, tf) #Very Slow!

plot_orbit_2D(position_arr)
plot_orbit_3D(position_arr)
