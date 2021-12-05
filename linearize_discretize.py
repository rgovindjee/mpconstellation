import numpy as np
from scipy import integrate


def linearize(f, x, u, tf, rho_func, drho_func, const, use_J2=False, use_drag=False):
    """
    Linearizes satellite dynamics about reference x and reference u
    gives A, B, Sigma, and xi (pg 22, pg 118)
    
    Args:
        f: Satellite dynamics function of the form dx = f(x,u)
        x: 7 vector, reference satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
        u: 3 vector, reference thrust [Tx, Ty, Tz]. ECI
        tf: scalar, refrence tf
        rho_func: Function to compute atmospheric density from r
        drho_func: Function to compute partial derivative of density w.r.t to r
        const: Constants object, with variables MU, R_E, J2, S, G0, ISP, CD
        use_J2: Boolean, determines if J2 is modeled. Default false.
        use_drag: Boolean, determines if drag is modeled. Default false.
    
    Returns:
        A: 7 x 7 matrix
        B: 7 x 3 matrix
        Sigma: 7 vector
        xi: 7 vector
    """
    # Extract position, velocity, mass from state vector
    r = np.vstack(x[0:3]) # 2D column vector, not 1D vector
    rx = x[0]
    ry = x[1]
    rz = x[2]
    r_norm = np.linalg.norm(r)
    v = np.vstack(x[3:6]) # 2D column vector, not 1D vector
    v_norm = np.linalg.norm(v)
    m = x[6]
    T = np.vstack(u) # 2D column vector, not 1D vector
    # Get atmospheric densities
    rho = rho_func(x[0:3])
    drho = drho_func(x[0:3])
    # Compute partial derivative terms according to appendix C.2
    # Partial derivative of a_g with respect to position
    Dr_ag = (-const.MU/(r_norm**3)*np.eye(3)
             + 3*const.MU/(r_norm**5)*np.dot(r, r.T))
    # Partial derivative of a_J2 with repspect to position
    if use_J2:
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
        Dr_aJ2 = np.zeros((3,3))
    # Partial derivative of a_D with respect to position, velocity, mass
    if use_drag:
        Dr_aD = np.matmul(-(const.CD*const.S/(2*m))*v_norm*v,
                        drho * r.T/r_norm)
        Dv_aD = -((rho*const.CD*const.S)/(2*m))*(v_norm*np.eye(3)
                + (1/(v_norm))*np.matmul(v, v.T))
        Dm_aD = ((rho*const.CD*const.S)/(2*m**2))*v_norm*v
    else:
        Dr_aD = np.zeros((3,3))
        Dv_aD = np.zeros((3,3))
        Dm_aD = np.zeros((3,1))
    # Partial derivative of a_T with respect to mass, thrust
    Dm_aT = -T/(m**2)
    DT_aT = (1/m)*np.eye(3)

    # Build Dxf
    Dxf = np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))]),
                     np.hstack([Dr_ag + Dr_aJ2 + Dr_aD, Dv_aD, Dm_aD + Dm_aT]),
                     np.zeros((1, 7))])

    # Build Duf
    DT_fm = -(T.T)/(const.G0*const.ISP*np.linalg.norm(T))
    Duf = np.vstack([np.zeros((3, 3)), DT_aT, DT_fm])

    # Calculate and output A, B, Sigma, xi
    A = tf*Dxf
    B = tf*Duf
    Sigma = f(x, u)
    xi = -(np.dot(A, x) + np.dot(B, u))
    return A, B, Sigma, xi


def discretize(f, x, u, tf, K, rho_func, drho_func, const):
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
    dtau = 1/(K-1)

    for k in range(0, K-1):
        tau_k = tau[k]
        tau_kp1 = tau[k+1]
        xk = x[:,k]
        uk = u[:,k]
        ukp1 = u[:,k+1]
        def u_func(tau):
            lambda_kn = (tau_kp1 - tau)/(tau_kp1 - tau_k)
            lambda_kp = (tau - tau_k)/(tau_kp1 - tau_k)
            return lambda_kn*uk + lambda_kp*ukp1
        sol = integrate.solve_ivp(f, [0, 1],)
        pass
    return None
        

# Helper functions