import numpy as np
from scipy import integrate
from datetime import datetime
from constants import *
from control import *
from satellite import Satellite

class Simulator:
    def __init__(self, sats=[], controller=Controller(), base_res=100):
        """
        Arguments:
            sats: list of Satellite objects to use for simulation
            controller = Controller object to use for simulation
        """
        self.sim_data = {}  # Data produced by simulator runs
        self.sim_full_state = {}  # Data produced by simulator runs
        self.sats = sats
        self.base_res = base_res  # Number of points evaluated in one orbit
        self.eval_points = self.base_res  # Initialize assuming one orbit
        self.controller = controller

    def run(self, tf=10):
        """
        Arguments:
            tf: rough number of orbits
        Runs a simulation for all satellites.
        Returns an dictionary with satellite IDs as keys and 3 x T arrays of x, y, z coordinates as values.
        """
        # Set resolution proportional to number of orbits
        self.eval_points = int(self.base_res*tf)
        pos_dict = {}
        state_dict = {}
        for sat in self.sats:
            u_func = self.controller.get_u_func()
            curr_pos = self.get_trajectory_ODE(sat, tf, u_func)
            # TODO(rgg): refactor so this is more elegant but also fix the test
            state_dict[sat.id] = self.last_run_state
            pos_dict[sat.id] = curr_pos
        self.sim_full_state = state_dict
        self.sim_data = pos_dict
        return self.sim_data

    @staticmethod
    def get_atmo_density(r, r0):
        """
        Arguments:
            r: current normalized position vector
            r0: initial position norm
        Returns:
            Atmospheric density at given altitude in kg/m^3
        Calculates atmospheric density based on current altitude and is
        only accurate between 480-520km because of linearization.
        Based on tabulated Harris-Priester Atmospheric Density Model found
        on page 91 of Satellite Orbits by Gill and Montenbruck
        """
        altitude = np.linalg.norm(r * r0) - R_EARTH
        # return 8E26 * altitude**-6.828 # power model for 400-600km but too slow
        # return -1E-17 * altitude**6E-12 # linear model for 480-520km - also slows down solver slightly
        return 9.983E-13 # fixed density for 500km


    @staticmethod
    def satellite_dynamics(tau, y, u_func, tf, const, include_drag=True, include_J2=True):
        """
        Arguments:
            tau: normalized time, values from 0 to 1
            y: state vector: [position, velocity, mass] - 7 x 1
            u_func: thrust function, u = u(y, tau).
            tf: final time used for normalization
            const: Constants class containing parameters MU, R_E, J2, S, G0, ISP
        Returns:
            difference to update state vector
        Dynamics function of the form y_dot = f(tau, y, params)
        """
        # References: (2.36 - 2.38, 2.95 - 2.97)
        # Get position, velocity
        r = y[0:3]
        v = y[3:6]
        # Get mass and thrust
        m = y[6]
        # TODO: investigate why this doesn't throw errors because y should be a 7-vec
        thrust = y[7:10]
        r_z = r[2]
        r_norm = np.linalg.norm(r)
        # Position ODE
        y_dot = np.zeros((7,))
        y_dot[0:3] = v # r_dot = v
        # Velocity ODE
        # Accel from gravity
        a_g = -const.MU/(r_norm)**3 * r
        # Accel from thrust; ignore thrust value
        a_u = u_func(y, tau) / m
        y_dot[3:6] = a_g + a_u
        if include_drag:
            # Accel from atmospheric drag
            a_d = -1/2 * C_D * const.S * (1 / m) * (Simulator.get_atmo_density(r, const.R0)/const.RHO) * np.linalg.norm(v) * v
            y_dot[3:6] += a_d
        if include_J2:
            # Accel from J2
            A = np.array([ [5*(r_z/r_norm)**2 - 1,0,0], [0,5*(r_z/r_norm)**2 - 1,0], [0,0,5*(r_z/r_norm)**2 - 3]])
            a_j2 = 1.5*const.J2*const.MU*const.R_E**2/np.linalg.norm(r)**5 * np.dot(A, r)
            y_dot[3:6] += a_J2
        # TODO(jx): implement accel from solar wind JX: No solar wind will be considered for now
        # Mass ODE
        y_dot[6] = -np.linalg.norm(thrust)/(const.G0*const.ISP)
        return tf*y_dot

    def get_trajectory_ODE(self, sat, tf, u_func):
        """
        Arguments:
            sat: Satellite object
            ts: timestep in seconds
            tf: (roughly) number of orbits, i.e. tf = 1 is 1 orbit.
            u_func: u(y, tau) takes in a normalized time tau and outputs a normalized 3x1 thrust vector.
        Returns:
            position: 3 x n array of x, y, z coordinates
        Get trajectory with ODE45, normalized dynamics.
        This function simulates the trajectory of a single satellite using the
        current state as the initial value.
        """
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
        const = Constants(MU=MU_EARTH/mu0, R_E=R_EARTH/r0, J2=J2, G0=G0/a0, ISP=ISP/s0, S=S/r0**2, R0=r0, RHO=m0/r0**3)

        # Solve IVP:
        sample_times = np.linspace(0, 1, self.eval_points) # Increase the number of samples as needed
        max_time_step = 0.001 # Adjust as needed for ODE accuracy
        sol = integrate.solve_ivp(Simulator.satellite_dynamics, [0, 1], y0, args=(u_func, tf, const), t_eval=sample_times, max_step=max_time_step)
        r = sol.y[0:3,:] # Extract positon vector
        # Save most recent full state data from simulation
        self.last_run_state = sol.y
        pos = r*r0 # Re-dimensionalize position [m]
        return pos

    def get_trajectory(self, sat, ts, tf):
        """
        Arguments:
            sat: Satellite object
            ts: timestep in seconds
            tf: final time in seconds
        Returns:
            position: 3 x n array of x, y, z coordinates
        Gets satellite trajectory with the forward Euler method
        """
        n = int(tf / ts)
        position = np.zeros(shape=(n, 3))
        init = sat.position
        for i in range(n):
            self.update_satellite_state(sat, ts)
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

    def save_to_csv(self, suffix=""):
        """
        Export trajectory to CSV for visualization in MATLAB
        """
        date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        for sat in self.sats:
            np.savetxt(f"trajectory_{date}_{sat.id}{suffix}.csv", self.sim_data[sat.id].T, delimiter=",")
