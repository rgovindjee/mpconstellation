import numpy as np
from scipy import integrate
from datetime import datetime
from constants import *
from control import *
from satellite import Satellite
from satellite_scale import SatelliteScale

class Simulator:
    def __init__(self, sats=[], controller=Controller(), scale=SatelliteScale(), base_res=100, include_drag = True, include_J2 = True):
        """
        Args:
            sats: list of Satellite objects to use for simulation
            controller = Controller object to use for simulation
            scale = SatelliteScale object, defines simulator output scaling.
        """
        self.sim_data = {}  # Data produced by simulator runs
        self.sim_time = {} # Time points for the simulator data
        self.sats = sats
        self.base_res = base_res  # Number of points evaluated in one orbit
        self.eval_points = self.base_res  # Initialize assuming one orbit
        self.controller = controller
        self.include_drag = include_drag
        self.include_J2 = include_J2
        self.scale = scale # Scaling object to dimensionalize, normalize

    def run(self, tf=10):
        """Runs a simulation for all satellites.
        Args:
            tf: rough number of orbits
        Returns: 
            A dictionary with satellite IDs as keys and 7 x T arrays of 
            state vectors, non-dimensionalzed (scaled) by simulator scale.
        """
        # Set resolution proportional to number of orbits
        self.eval_points = int(self.base_res*tf)
        state_dict = {}
        time_dict = {}
        for sat in self.sats:
            u_func = self.controller.get_u_func()
            sol = self.get_trajectory_ODE(sat, tf, u_func)
            state_dict[sat.id] = sol.y
            time_dict[sat.id] = sol.t
        self.sim_data = state_dict
        self.sim_time = time_dict
        return self.sim_data, self.sim_time

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
        # Get position, velocity, mass
        r = y[0:3]
        v = y[3:6]
        m = y[6]
        # Define helper variables
        r_z = r[2]
        r_norm = np.linalg.norm(r)
        # Position ODE
        y_dot = np.zeros((7,))
        y_dot[0:3] = v # r_dot = v
        # Velocity ODE
        # Accel from gravity
        a_g = -const.MU/(r_norm)**3 * r
        # Accel from thrust; ignore thrust value
        u = u_func(y, tau) # Get thrust
        a_u = u / m
        y_dot[3:6] = a_g + a_u
        if include_drag:
            # Accel from atmospheric drag
            a_d = -1/2 * C_D * const.S * (1 / m) * (Simulator.get_atmo_density(r, const.R0)/const.RHO) * np.linalg.norm(v) * v
            y_dot[3:6] += a_d
        if include_J2:
            # Accel from J2
            A = np.array([ [5*(r_z/r_norm)**2 - 1,0,0], [0,5*(r_z/r_norm)**2 - 1,0], [0,0,5*(r_z/r_norm)**2 - 3]])
            a_J2 = 1.5*const.J2*const.MU*const.R_E**2/np.linalg.norm(r)**5 * np.dot(A, r)
            y_dot[3:6] += a_J2
        # Mass ODE
        y_dot[6] = -np.linalg.norm(u)/(const.G0*const.ISP)
        return tf*y_dot


    def get_trajectory_ODE(self, sat, tf, u_func):
        """
        Arguments:
            sat: Satellite object
            ts: timestep in seconds
            tf: (roughly) number of orbits, i.e. tf = 1 is 1 orbit.
            u_func: u(x, tau) takes in state vector x and a normalized time 
                    tau and outputs a normalized 3x1 thrust vector.
        Returns:
            state: 7 x n array of state vectors
        Get trajectory with ODE45, normalized dynamics.
        This function simulates the trajectory of a single satellite using the
        current state as the initial value.
        """
        # Normalized state vector (pg. 21)
        y0 = self.scale.normalize_state(sat.get_state_vector())

        # Normalize system parameters (pg. 21)
        const = self.scale.get_normalized_constants()

        # Solve IVP:
        sample_times = np.linspace(0, 1, self.eval_points) # Increase the number of samples as needed
        max_time_step = 0.001 # Adjust as needed for ODE accuracy
        sol = integrate.solve_ivp(Simulator.satellite_dynamics, [0, 1], y0, args=(u_func, tf, const, self.include_drag, self.include_J2), t_eval=sample_times, max_step=max_time_step)
        # Output solution from solve_ivp
        return sol


    def save_to_csv(self, suffix="", redimensionalize = True):
        """
        Export trajectory to CSV for visualization in MATLAB
        """
        date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        for sat in self.sats:
            if redimensionalize:
                np.savetxt(f"trajectory_{date}_{sat.id}{suffix}.csv", self.scale.redim_state(self.sim_data[sat.id]).T, delimiter=",")
            else:
                np.savetxt(f"trajectory_{date}_{sat.id}{suffix}.csv", self.sim_data[sat.id].T, delimiter=",")
