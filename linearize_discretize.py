import numpy as np
from scipy import integrate, interpolate
from simulator import Simulator
import logging
from functools import partial
import multiprocessing as mp


def get_matrices(options,funcs, tf, tau, x, k):
    """
    Options:
        'use_uniform_steps'
        'integrator_steps'
        'ivp_max_step'
        'ivp_solver'
    Funcs:
        'dPhi_gen'
        'f'
        'u_func'
        'B_func'
        'Sigma_func'
        'xi_func'
    """
    # Ideally make the for loop below parallelized
    # Extract values
    tau_k = tau[k]  # Left bound of temporal node
    tau_kp1 = tau[k+1]  # Right bound of temporal node
    if options['use_uniform_steps']:
        tau_points = np.linspace(tau_k, tau_kp1, options['integrator_steps']) # Set times for which solve_ivp should output values
    else:
        tau_points = None # No pre-determined evaluation times, let solve_ivp() decide
    x_k = x[:, k]  # Get reference state for current node
    # Solve for the state transition matrix Phi, evaluated at tau_points
    # Define initial value for integrating
    y0 = np.concatenate([np.eye(7).flatten(), x_k])
    # Numerically integrate to solve for the state transition matrix Phi
    dPhi = funcs['dPhi_gen']()
    sol = integrate.solve_ivp(dPhi, [tau_k, tau_kp1], y0,
                                args=(funcs['f'], funcs['u_func'], tf),
                                max_step=options['ivp_max_step'],
                                method=options['ivp_solver'],
                                t_eval=tau_points)
    # Extract final phi to get equation A_k = Phi(k+1)
    Phi_kp1 = np.reshape(sol.y[0:49, -1], (7, 7))
    A_k = Phi_kp1 # Store in A_k array

    # Numerically integrate for Bk-, Bk+, Sigma, xi
    if options['use_uniform_steps']:
        int_points = tau_points
    else:
        int_points = sol.t
    # Extract a series of Phi matrices
    Phi_series = np.reshape(sol.y[0:49,:].T, (int_points.size, 7,7))
    # Extract state vectors
    x_series = sol.y[49:56,:]
    # Preallocate arrays
    B_series = np.zeros((int_points.size, 7,3))
    Sigma_series = np.zeros((7,int_points.size))
    xi_series = np.zeros((7, int_points.size))
    # Calculate lambda terms (for use in B integrand)
    lambda_kn = (tau_kp1 - int_points)/(tau_kp1 - tau_k)
    lambda_kp = (int_points - tau_k)/(tau_kp1 - tau_k)
    # Evaluate B, Sigma, and xi linearization functions
    for i, t in enumerate(int_points):
        # Call u_func with None for states since they aren't used
        B_series[i,:,:] = funcs['B_func'](x_series[:,i],funcs['u_func'](None, t),tf)
        Sigma_series[:,i] = funcs['Sigma_func'](funcs['f'], x_series[:,i], funcs['u_func'], tau=t)
        xi_series[:,i] = funcs['xi_func'](funcs['f'], x_series[:,i], funcs['u_func'](None, t),tf)

    Phi_inv = np.linalg.inv(Phi_series) # Phi must be invertible
    # Compute B integrands, size is n x 3 x 3
    Bn_integrand = Phi_inv @ (B_series * lambda_kn[:,None,None])
    Bp_integrand = Phi_inv @ (B_series * lambda_kp[:,None,None])
    # Compute Sigma, xi integrands, size is n x 7
    Sigma_integrand = np.column_stack([Phi_inv[i,:,:] @ Sigma_series[:,i] for i in range(0,int_points.size)])
    xi_integrand = np.column_stack([Phi_inv[i,:,:] @ xi_series[:,i] for i in range(0,int_points.size)])
    # Numerically integrate with trapz, along the right axis, and store
    B_kp = (Phi_kp1 @ np.trapz(y = Bp_integrand, x = int_points, axis = 0))
    B_kn = (Phi_kp1 @ np.trapz(y = Bn_integrand, x = int_points, axis = 0))
    Sigma_k = (Phi_kp1 @ np.trapz(y = Sigma_integrand, x = int_points, axis = 1))
    xi_k = (Phi_kp1 @ np.trapz(y = xi_integrand, x = int_points, axis = 1))

    return [A_k, B_kp, B_kn, Sigma_k, xi_k]


class Discretizer():
    def __init__(self, const, rho_func=Simulator.get_atmo_density, drho_func=None, include_drag=False, include_J2=False, use_scipy_ZOH = False):
        """
        Args:
            const: Constants object, with variables MU, R_E, J2, S, G0, ISP, CD
        """
        self.const = const

        # Global variables that affect dynamics
        self.include_drag = include_drag
        self.include_J2 = include_J2
        self.use_scipy_ZOH = use_scipy_ZOH

        # TODO: Chase to look at rho_func and drho_func
        self.rho_func = rho_func # Not necessarily the right function
        self.drho_func = drho_func

        # ODE solver parameters
        self.ivp_max_step = 1e-2
        self.ivp_solver = 'RK45'

        # Numerical integration parameters
        self.integrator_steps = 101
        self.use_uniform_steps = False # If False, integration ignores self.integrator_steps and uses non-uniform steps from solve_ivp()

        # Set up logging
        logging.basicConfig(level=logging.WARNING)

        # Store x or u as needed
        self.__u = None
        self.__tau = None


    def A_func(self, x, u, tf):
        """
        Linearize satellite dynamics about reference x and reference u
        gives A (pg 22, pg 118)

        Args:
            f: Satellite dynamics function of the form dx = f(x,u)
            x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
            u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI
            tf: scalar, reference tf

        Returns:
            A: 7 x 7 matrix
        """
        # Extract position, velocity, mass from state vector
        r = np.vstack(x[0:3])  # 2D column vector, not 1D vector
        rx = x[0]
        ry = x[1]
        rz = x[2]
        r_norm = np.linalg.norm(r)
        logging.debug(f"x: {x}")
        v = np.vstack(x[3:6])  # 2D column vector, not 1D vector
        v_norm = np.linalg.norm(v)
        m = x[6]
        T = np.vstack(u)  # 2D column vector, not 1D vector
        # Compute partial derivative terms according to appendix C.2
        # Partial derivative of a_g with respect to position
        Dr_ag = (-self.const.MU/(r_norm**3)*np.eye(3)
                 + 3*self.const.MU/(r_norm**5)*np.dot(r, r.T))
        # Partial derivative of a_J2 with repspect to position
        if self.include_J2:
            kJ2 = 1.5*self.const.J2*self.const.MU*self.const.R_E**2
            rz_norm_sq = (rz/r_norm)**2
            GJ2 = np.diag([5*rz_norm_sq-1, 5*rz_norm_sq-1, 5*rz_norm_sq-3])
            ddr = (5*(rz**2)*(-2*(r.T/(r_norm**4)))
                   + (5/(r_norm**2))*np.array([[0, 0, 2*rz]]))
            Dr_aJ2 = ((kJ2*GJ2 @ r) @ (-5*r.T/(r_norm**7))
                      + kJ2/(r_norm**5)
                      * np.vstack([rx*ddr, ry*ddr, rz*ddr])
                      + kJ2/(r_norm**5) * (GJ2 @ np.eye(3)))
        else:
            Dr_aJ2 = np.zeros((3, 3))
        # Partial derivative of a_D with respect to position, velocity, mass
        if self.include_drag:
            # Get atmospheric densities
            rho = self.rho_func(x[0:3])
            drho = self.drho_func(x[0:3])
            Dr_aD = ((-self.const.CD*self.const.S/(2*m))*v_norm*v)@(drho * r.T/r_norm)
            Dv_aD = ((-rho*self.const.CD*self.const.S)/(2*m))*(v_norm*np.eye(3)
                    + (1/(v_norm))*(v @ v.T))
            Dm_aD = ((rho*self.const.CD*self.const.S)/(2*m**2))*v_norm*v
        else:
            Dr_aD = np.zeros((3, 3))
            Dv_aD = np.zeros((3, 3))
            Dm_aD = np.zeros((3, 1))
        # Partial derivative of a_T with respect to mass
        Dm_aT = -T/(m**2)
        # Build Dxf
        Dxf = np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))]),
                         np.hstack([Dr_ag + Dr_aJ2 + Dr_aD, Dv_aD, Dm_aD + Dm_aT]),
                         np.zeros((1, 7))])

        # Calculate and output A
        A = tf*Dxf
        return A


    def B_func(self, x, u, tf):
        """
        Linearize satellite dynamics about reference x and reference u
        gives B (pg 22, pg 118)

        Args:
            x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
            u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI
            tf: scalar, refrence tf

        Returns:
            B: 7 x 3 matrix
        """
        # Extract mass from state vector
        m = x[6]
        T = np.vstack(u)  # 2D column vector, not 1D vector
        # Partial derivative of a_T with respect to thrust
        DT_aT = (1/m)*np.eye(3)
        # Build Duf
        logging.debug(f"B_func T:\n{T}")
        logging.debug(f"norm T: {np.linalg.norm(T)}")
        norm_T = np.linalg.norm(T)
        if norm_T <= np.finfo(float).eps:
            DT_fm = np.array([[0., 0., 0.]])
        else:
            DT_fm = -(T.T)/(self.const.G0*self.const.ISP*norm_T)
        Duf = np.vstack([np.zeros((3, 3)), DT_aT, DT_fm])
        # Calculate and output B
        B = tf*Duf
        return B


    def xi_func(self, f, x, u, tf):
        """
        Linearize satellite dynamics about reference x and reference u, gives xi

        Args:
            f: Satellite dynamics function of the form dx = f(x,u)
            x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
            u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI
            tf: scalar, refrence tf

        Returns:
            xi: 7 vector
        """
        # Compute A and B matrices
        A = self.A_func(x, u, tf)
        B = self.B_func(x, u, tf)
        # Compute xi (pg 22)
        xi = -((A @ x) + (B @ u))
        return xi


    def Sigma_func(self, f, x, u_func, tau):
        """
        Linearize satellite dynamics about reference x and reference u, gives Sigma

        Args:
            f: Satellite dynamics function of the form dx = f(x,u)
            x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
            u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI

        Returns:
            Sigma: 7 vector
        """
        # Compute Sigma (pg 22)
        tf = 1 # tf must equal 1, Sigma is the non-normalized dynamics
        Sigma = f(tau, x, u_func, tf, self.const, include_J2 = self.include_J2, include_drag = self.include_drag)
        return Sigma


    def dPhi_gen(self):
        """
        Generates a function that generates the matrix differential equation for the state transition matrix Phi
        Necessary because solve_ivp doesn't handle class methods well(?)
        """
        def dPhi(tau, y, f, u_func, tf):
            """
            Args:
                tau: Current time tau
                y: Diff eq. vector, size 56. First 49 elements for the individual
                   entries in Phi. Remaining 7 elements is the state vector x
                f: Full nonlinear dynamics function for the satellite
                u_func: Function to calculate u from tau
                tf: Reference final time, used in scaling
            """

            # y is the combine state vector of phi and x of size (49+7,0)
            # Calculate u with ufunc, using None for states since interpolating
            u = u_func(None, tau)
            # Extract Phi and x
            Phi = np.reshape(y[0:49], (7, 7))
            logging.debug(f"y shape: {y.shape}")
            x = y[49:56]
            logging.debug(f"x shape: {x.shape}")
            # Calculate A
            A = self.A_func(x, u, tf)
            # Update new Phi and x
            Phi_dot = A @ Phi
            # TODO: Investigate if tf is supposed to be in f
            # RESOLVED: (Jason) I believe tf is supposed to be in f
            x_dot = f(tau, x, u_func, tf, self.const, include_J2 = self.include_J2, include_drag = self.include_drag)
            # Flatten phi and store back into new vector
            y_dot = np.concatenate([Phi_dot.flatten(), x_dot])
            return y_dot
        return dPhi


    def u_FOH(self, tau, u):
        """
        First order hold interpolation of a signal u

        Args:
            tau: Current time tau
            u: n x K matrix of discrete inputs

        Returns:
            Interpolated u at time tau, u(tau)
        """
        if tau == 1:
            return u[:,-1]
        else:
            K = u.shape[1] # Number of points for discretization
            dtau = 1/(K-1) # Length of each interval in tau units
            k = int(tau // dtau) # lower index of interval to interpolate in
            tau_k = k/(K-1) # left bound of interval
            tau_kp1 = (k+1)/(K-1) # right bound of interval
            lambda_kn = (tau_kp1 - tau)/(tau_kp1 - tau_k)
            lambda_kp = (tau - tau_k)/(tau_kp1 - tau_k)
            return lambda_kn*u[:, k] + lambda_kp*u[:, k+1]


    def u_func(self, x, t):
        """
        u_func of standard form
        Args:
            x: state vector at time t
            t: time
        Returns:
            u: (3,) thrust vector
        """
        if self.use_scipy_ZOH:
            # takes slightly longer
            return interpolate.interp1d(self.__tau, self.__u, kind='linear', axis = 1)(t)
        else:
            return self.u_FOH(t, self.__u)


    def discretize(self, f, x, u, tf):
        """
        Discretizes and linearizes satellite dynamics
        with K temporal nodes from tau = [0, 1]

        Args:
            f: x_dot = f(x, u)
            x: 7 x K reference trajectory
            u: 3 x K reference input (thrust)
            tf: scalar, reference tf
            K: integer, number of temporal nodes

        Returns:
            A_k: 3D numpy array of shape (K-1, 7, 7). Each slice is A_k at a time k
            B_kp: 3D numpy array of shape (K-1, 7, 3).
            B_kn: 3D numpy array of shape (K-1, 7, 3).
            Sigma_k: 2D numpy array of shape (7, K-1).
            xi_k: 2D numpy array of shape (7, K-1).
        """
        # Infer K from reference trajectory
        K = x.shape[1]
        #
        tau = np.linspace(0, 1, K)
        self.__tau = tau
        self.__u = u
        # Preallocate Output Arrays
        A_k = np.zeros((K-1,7,7))
        B_kp = np.zeros((K-1, 7, 3))
        B_kn = np.zeros((K-1, 7, 3))
        Sigma_k = np.zeros((7, K-1))
        xi_k = np.zeros((7, K-1))

        # Output arrays of linearization matrices
        options={'use_uniform_steps':self.use_uniform_steps,
                'integrator_steps':self.integrator_steps,
                'ivp_max_step':self.ivp_max_step,
                'ivp_solver':self.ivp_solver}
        funcs={'dPhi_gen':self.dPhi_gen,
                'f':f, 'u_func':self.u_func, 'B_func': self.B_func, 'Sigma_func': self.Sigma_func,
                'xi_func': self.xi_func}

        g = partial(get_matrices,options, funcs, tf, tau, x)

        pool = mp.Pool(mp.cpu_count())
        result = pool.map(g, range(0, K-1))
        pool.close()
        pool.join()

        # Concatenate pool results into expected values
        for i, r in enumerate(result):
            A_k[i,:,:] = r[0]
            B_kp[i,:,:] = r[1]
            B_kn[i,:,:] = r[2]
            Sigma_k[:,i] = r[3]
            xi_k[:,i] = r[4]

        return A_k, B_kp, B_kn, Sigma_k, xi_k


    @staticmethod
    def extract_uk(x_k, tau_k, controller):
        """Utility function to extract u_k from x_k and tau_k
        Extracts the control inputs u_k needed for linearize/discretize based
        on guess states of x_k at tau_k as outputted by the simulator.

        Args:
            x_k: 7 x K numpy array of state vectors
            tau_k: numpy vector of size K of sample times
            controller: same controller as used by simulator to generate x_k

        Returns:
            3 x K numpy array of control inputs u
        """
        u_func = controller.get_u_func()
        u_k = []
        for i in range(x_k.shape[1]): # Iterate through each column
            u_k.append(u_func(x_k[:,i],tau_k[i]))
        return np.column_stack(u_k)
