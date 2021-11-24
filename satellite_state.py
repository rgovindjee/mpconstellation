import numpy as np
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from constants import *

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

def satellite_dynamics(tau, y, u, tf, constants):
    # Dynamics function of the form y_dot = f(tau, y, params)
    # tau is the normalized time, values from 0 to 1
    # y is the state vector: [position, velocity, mass] - 7 x 1
    # u is the thrust function, u = u(t)
    # tf is the final time used for normalization
    # constants is a dict, containing keys MU, R_E, J2, S, G0, ISP

    # References: (2.36 - 2.38, 2.95 - 2.97)
    # Get position, velocity
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    thrust = y[7:10]
    r_z = r[2]
    r_norm = np.linalg.norm(r)
    # Position ODE
    y_dot = np.zeros((7,))
    y_dot[0:3] = v # r_dot = v
    # Velocity ODE
    # Accel from gravity
    a_g = -constants['MU']/(r_norm)**3 * r
    # Accel from J2
    A = np.array([ [5*(r_z/r_norm)**2 - 1,0,0], [0,5*(r_z/r_norm)**2 - 1,0], [0,0,5*(r_z/r_norm)**2 - 3]])
    a_j2 = 1.5*constants['J2']*constants['MU']*constants['R_E']**2/np.linalg.norm(r)**5 * np.dot(A, r)
    # Accel from drag (To Be Implemented)
    # Accel from solar wind (To Be Implemented)
    y_dot[3:6] = a_g + a_j2
    # Mass ODE
    y_dot[6] = -np.linalg.norm(thrust)/(constants['G0']*constants['ISP'])

    return tf*y_dot

# Forward Euler time-step method
def update_satellite_state(sat, time_step):
    # Update Position
    position_next = sat.velocity * time_step + sat.position

    # Update Velocity
    accel = -(MU_EARTH / np.linalg.norm(sat.position) ** 3) * sat.position + sat.thrust / sat.mass
    velocity_next = accel * time_step + sat.velocity

    # Update Mass
    mass_dot = np.linalg.norm(sat.thrust) / G0 * ISP
    mass_next = mass_dot * time_step + sat.mass

    # Update Satellite object
    sat.position = position_next
    sat.velocity = velocity_next
    sat.mass = mass_next

# Get trajectory with ODE45, normalized dynamics
def get_trajectory_ODE(sat, tf):
    # This function simulates the trajectory of the satellite using the current state as the initial value
    # tf roughly corresponds to number of orbits, ie tf = 1 is 1 orbit
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
    const = {'MU': MU_EARTH/mu0, 'R_E': R_EARTH/r0, 'J2': J2, 'S':0, 'G0':G0/a0, 'ISP':ISP/s0}

    # Arbitrary zero thrust input
    u = lambda tau: np.array([0,0,0])

    # Solve IVP:
    resolution = (100*tf) + 1 # Generally, higher resolution for more orbits are needed
    times = np.linspace(0, 1, resolution)
    sol = integrate.solve_ivp(satellite_dynamics, [0, tf], y0, args=(u, tf, const), t_eval=times, max_step=0.001)
    r = sol.y[0:3,:] # Extract positon vector
    pos = r*r0 # Re-dimensionalize position [m]
    return pos

# Get trajectory with the forward euler method
def get_trajectory(sat, ts, tf):
    n = int(tf / ts)
    position = np.zeros(shape=(n, 3))
    init = sat.position
    for i in range(n):
        update_satellite_state(sat, ts)
        position[i, :] = sat.position
        if np.linalg.norm(position[i, :]) < R_EARTH:
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
    earth = plt.Circle((0, 0), R_EARTH, color='g')
    ax.add_patch(earth)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_orbit_3D(position_arr):
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


# Not Used
# altitude_km = 5000  # m
# height = altitude_km + R_EARTH  # m

# Initial states are based on orbit of Hubble Space Telescope on January 19, 2016
# Initial position
r0 = np.array([5371.4806, -4133.1393, 1399.9594]) * 1000  # m
# Initial velocity
v0 = np.array([4.6921, 4.9848, -3.2752]) * 1000 # m/s
# Initial Thrust
T0 = np.array([0, 0, 0])  # N
# Initial Mass
m0 = 12200  # kg


pos = get_trajectory_ODE(Satellite(r0, v0, m0, T0))
np.savetxt("trajectory.csv", pos.T, delimiter=",") # Export trajectory, plot in MATLAB

# Time
#ts = .01  # Time step in [s]
#tf_hours = 3.25  # Final time in [hr]
#tf = tf_hours * 3600  # s

# get_trajectory(Satellite(r0, v0, m0, T0), 10, tf)
# get_trajectory(Satellite(r0, v0, m0, T0), 1, tf)
#position_arr = get_trajectory(Satellite(r0, v0, m0, T0), .1, tf)
# get_trajectory(Satellite(r0, v0, m0, T0), .01, tf) #Very Slow!

#plot_orbit_2D(position_arr)
#plot_orbit_3D(position_arr)
