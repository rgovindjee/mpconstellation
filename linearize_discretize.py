import numpy as np
from scipy import integrate
from simulator import Simulator

# Global variables that affect dynamics
include_drag = False
include_J2 = False
rho_func = Simulator.get_atmo_density
drho_func = None


def A_func(f, x, u, tf, const):
    """
    Linearize satellite dynamics about reference x and reference u
    gives A (pg 22, pg 118)

    Args:
        f: Satellite dynamics function of the form dx = f(x,u)
        x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
        u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI
        tf: scalar, refrence tf
        const: Constants object, with variables MU, R_E, J2, S, G0, ISP, CD

    Returns:
        A: 7 x 7 matrix
    """
    # Extract position, velocity, mass from state vector
    r = np.vstack(x[0:3])  # 2D column vector, not 1D vector
    rx = x[0]
    ry = x[1]
    rz = x[2]
    r_norm = np.linalg.norm(r)
    v = np.vstack(x[3:6])  # 2D column vector, not 1D vector
    v_norm = np.linalg.norm(v)
    m = x[6]
    T = np.vstack(u)  # 2D column vector, not 1D vector
    # Get atmospheric densities
    rho = rho_func(x[0:3])
    drho = drho_func(x[0:3])
    # Compute partial derivative terms according to appendix C.2
    # Partial derivative of a_g with respect to position
    Dr_ag = (-const.MU/(r_norm**3)*np.eye(3)
             + 3*const.MU/(r_norm**5)*np.dot(r, r.T))
    # Partial derivative of a_J2 with repspect to position
    if include_J2:
        kJ2 = 1.5*const.J2*const.MU*const.R_E**2
        rz_norm_sq = (rz/r_norm)**2
        GJ2 = np.diag([5*rz_norm_sq-1, 5*rz_norm_sq-1, 5*rz_norm_sq-3])
        ddr = (5*(rz**2)*(-2*(r.T/(r_norm**4)))
               + (5/(r_norm**2))*np.array([[0, 0, 2*rz]]))
        Dr_aJ2 = (np.matmul(kJ2*GJ2*r, -5*r.T/(r_norm**7))
                  + kJ2/(r_norm**5)
                  * np.vstack([rx*ddr, ry*ddr, rz*ddr])
                  + kJ2/(r_norm**5) * GJ2 * np.eye(3))
    else:
        Dr_aJ2 = np.zeros((3, 3))
    # Partial derivative of a_D with respect to position, velocity, mass
    if include_drag:
        Dr_aD = np.matmul(-(const.CD*const.S/(2*m))*v_norm*v,
                          drho * r.T/r_norm)
        Dv_aD = -((rho*const.CD*const.S)/(2*m))*(v_norm*np.eye(3)
                                                 + (1/(v_norm))*np.matmul(v, v.T))
        Dm_aD = ((rho*const.CD*const.S)/(2*m**2))*v_norm*v
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


def B_func(x, u, tf, const):
    """
    Linearize satellite dynamics about reference x and reference u
    gives B (pg 22, pg 118)

    Args:
        x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
        u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI
        tf: scalar, refrence tf
        const: Constants object, with variables MU, R_E, J2, S, G0, ISP, CD

    Returns:
        B: 7 x 3 matrix
    """
    # Extract mass from state vector
    m = x[6]
    T = np.vstack(u)  # 2D column vector, not 1D vector
    # Partial derivative of a_T with respect to thrust
    DT_aT = (1/m)*np.eye(3)
    # Build Duf
    DT_fm = -(T.T)/(const.G0*const.ISP*np.linalg.norm(T))
    Duf = np.vstack([np.zeros((3, 3)), DT_aT, DT_fm])
    # Calculate and output B
    B = tf*Duf
    return B


def xi_func(f, x, u, tf, const):
    """
    Linearize satellite dynamics about reference x and reference u, gives xi

    Args:
        f: Satellite dynamics function of the form dx = f(x,u)
        x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
        u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI
        tf: scalar, refrence tf
        const: Constants object, with variables MU, R_E, J2, S, G0, ISP, CD

    Returns:
        xi: 7 vector
    """
    # Compute A and B matrices
    A = A_func(f, x, u, tf, const)
    B = B_func(x, u, tf, const)
    # Compute xi (pg 22)
    xi = -(np.dot(A, x) + np.dot(B, u))
    return xi


def Sigma_func(f, x, u):
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
    Sigma = f(x, u)
    return Sigma


def dPhi(tau, y, f, u_func, tf, const):
    """
    Matrix differential equation for the state transition matrix Phi

    Args:
        tau: Current time tau
        y: Diff eq. vector, size 56. First 49 elements for the individual
           entries in Phi. Remaining 7 elements is the state vector x
        f: Full nonlinear dynamics function for the satellite
        u_func: Function to calculate u from tau
        tf: Reference final time, used in scaling
        const: Constants object, with variables MU, R_E, J2, S, G0, ISP, CD
    """
    # y is the combine state vector of phi and x of size (49+7,0)
    # Calculate u with ufunc
    u = u_func(tau)
    # Extract Phi and x
    Phi = np.reshape(y[0:49], (7, 7))
    x = y[49:56]
    # Calculate A
    A = A_func(f, x, u, tf, const)
    # Update new Phi and x
    Phi_dot = A @ Phi
    x_dot = f(x, u)
    # Flatten phi and store back into new vector
    y_dot = np.concatenate([Phi_dot.flatten(), x_dot])
    return y_dot

def u_FOH(tau, u):
    """
    First order hold interpolation of a signal u

    Args:
        tau: Current time tau
        u: n x K matrix of discrete inputs
    
    Returns:
        Interpolated u at time tau, u(tau)
    """
    K = u.size
    dtau = 1/(K-1)
    k = tau // dtau
    tau_k = (k - 1)/(K-1)
    tau_kp1 = (k)/(K-1)
    lambda_kn = (tau_kp1 - tau)/(tau_kp1 - tau_k)
    lambda_kp = (tau - tau_k)/(tau_kp1 - tau_k)
    return lambda_kn*u[k] + lambda_kp*u[k+1]


def discretize(f, x, u, tf, K, const):
    """
    Discretizes and linearizes satellite dynamics
    with K temporal nodes from tau = [0, 1]

    Args:
        x: 7 x K reference trajectory
        u: 3 x K reference input (thrust)
        tf: scalar, reference tf
        K: integer, number of temporal nodes

    Returns:
        A_k: List of length k, each element is a 7 x 7 matrix
        B_kp: List of length k, each element is a 7 x 3 matrix
        B_kn:  List of length k, each element is a 7 x 3 matrix
        Sigma_k: List of length k, each element is a 7 vector
        xi_k: List of length k, each element is a 7 vector

    """
    tau = np.linspace(0, 1, K)
    A_k = []
    B_kp = []
    B_kn = []
    Sigma_k = []
    xi_k = []

    u_func = lambda tau: u_FOH(tau, u)

    # Ideally make the for loop below parallelized
    for k in range(0, K-1):
        # Extract values
        tau_k = tau[k]  # Left bound of temporal node
        tau_kp1 = tau[k+1]  # Right bound of temporal node
        tau_points = np.linspace(tau_k, tau_kp1, 101)  # Used for intergration
        x_k = x[:, k]  # Get reference state for current node

        # Solve for the state transition matrix Phi, evaluated at tau_points
        # Define initial value for integrating
        y0 = np.concatenate([np.eye(7).flatten(), x_k])
        # Numerically integrate to solve for the state transition matrix Phi
        sol = integrate.solve_ivp(dPhi, [tau_k, tau_kp1], y0,
                                  args=(f, u_func, tf, const),
                                  max_step=1e-4, t_eval=tau_points)
        # Extract final phi to get equation A_k = Phi(k+1)
        Phi_kp1 = np.reshape(sol.y[0:49, -1], (7, 7))
        A_k.append(Phi_kp1)
        
        # Numerically integrate for Bk-, Bk+, Sigma, xi
        Phi_series = np.reshape(sol.y[0:49,:], (tau_points.size, 7,7))
        x_series = sol.y[49:56, :]
        B_series = np.zeros((tau_points.size, 7,3))
        Sigma_series = np.zeros((7,tau_points.size))
        xi_series = np.zeros((7, tau_points.size))
        lambda_kn = (tau_kp1 - tau_points)/(tau_kp1 - tau_k)
        lambda_kp = (tau_points - tau_k)/(tau_kp1 - tau_k)
        for i, t in enumerate(tau_points):
            B_series[i,:,:] = B_func(x_series[:,i],u_func(t),tf,const)
            Sigma_series[:,i] = Sigma_func(f, x_series[:,i], u_func(t))
            xi_series[:,i] = xi_func(f, x_series[:,i], u_func(t),tf, const)

        Phi_inv = np.linalg.inv(Phi_series)
        Bn_integrand = Phi_inv @ (B_series * lambda_kn[:,None,None])
        Bp_integrand = Phi_inv @ (B_series * lambda_kp[:,None,None])
        Sigma_integrand = np.column_stack([Phi_inv[i,:,:] @ Sigma_series[:,i] for i in range(0,tau_points.size)])
        xi_integrand = np.column_stack([Phi_inv[i,:,:] @ xi_series[:,i] for i in range(0,tau_points.size)])
        B_kp.append(Phi_kp1 @ np.trapz(y = Bp_integrand, x = tau_points, axis = 0))
        B_kn.append(Phi_kp1 @ np.trapz(y = Bn_integrand, x = tau_points, axis = 0))
        Sigma_k.append(Phi_kp1 @ np.trapz(y = Sigma_integrand, x = tau_points, axis = 1))
        xi_k.append(Phi_kp1 @ np.trapz(y = xi_integrand, x = tau_points, axis = 1))

    return A_k, B_kp, B_kn, Sigma_k, xi_k
