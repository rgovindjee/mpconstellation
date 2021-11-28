import numpy as np
from scipy import integrate
from constants import *
from satellite import Satellite

class Simulator:
    def __init__(self):
        self.sim_data = np.array([])  # Data produced by simulator runs
        return
    @staticmethod
    def satellite_dynamics(tau, y, u, tf, constants):
        """
        Arguments:
            tau: normalized time, values from 0 to 1
            y: state vector: [position, velocity, mass] - 7 x 1
            u: thrust function, u = u(t)
            tf: final time used for normalization
            constants: dict, containing keys MU, R_E, J2, S, G0, ISP
        Returns:
            difference to update state vector
        Dynamics function of the form y_dot = f(tau, y, params)
        """
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
        # TODO(jx): implement accel from drag
        # TODO(jx): implement accel from solar wind
        y_dot[3:6] = a_g + a_j2
        # Mass ODE
        y_dot[6] = -np.linalg.norm(thrust)/(constants['G0']*constants['ISP'])
        return tf*y_dot

    def get_trajectory_ODE(self, sat, tf):
        """
        Arguments:
            sat: Satellite object
            ts: timestep in seconds
            tf: (roughly) number of orbits, i.e. tf = 1 is 1 orbit.
        Returns:
            position: nx3 array of x, y, z coordinates
        Get trajectory with ODE45, normalized dynamics.
        This function simulates the trajectory of the satellite using the 
        current state as the initial value.
        Designer units (pg. 20)
        """
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
        sol = integrate.solve_ivp(Simulator.satellite_dynamics, [0, tf], y0, args=(u, tf, const), t_eval=times, max_step=0.001)
        r = sol.y[0:3,:] # Extract positon vector
        pos = r*r0 # Re-dimensionalize position [m]
        self.sim_data = pos
        return pos

    # Get trajectory with the forward euler method
    def get_trajectory(self, sat, ts, tf):
        """
        Arguments:
            sat: Satellite object
            ts: timestep in seconds
            tf: final time in seconds
        Returns:
            position: nx3 array of x, y, z coordinates
        """
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
        self.sim_data = position
        return position

    # Forward Euler time-step method
    @staticmethod
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

    def save_to_csv(self):
        """
        Export trajectory to CSV for visualization in MATLAB
        """
        np.savetxt("trajectory.csv", self.sim_data.T, delimiter=",")