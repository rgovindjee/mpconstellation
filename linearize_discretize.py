import numpy as np
import scipy.signal


def linearize(f, x, u, tf, rho_func, drho_func, const, use_J2=False, use_drag=False):
    """Linearizes satellite dynamics about reference x and reference u
       gives A, B, Sigma, and xi (pg 22, pg 118)
    Parameters:
    f: Satellite dynamics function of the form x = f(x,u)
    x: Vector of size 7, satellite states [rx, ry, rz, vx, vy, vz, mass]. ECI
    u: Vector of size 3, thrust [Tx, Ty, Tz]. ECI
    rho: Function to compute atmospheric density from r
    drho: Function to compute partial derivative of density w.r.t to r
    constants: dict, containing keys MU, R_E, J2, S, G0, ISP, CD, S
    use_J2: Boolean, determines if J2 is modeled. Default false.
    use_drag: Boolean, determines if drag is modeled. Default false.
    Returns:
    A: Matrix of size 7 x 7
    B: matrix of size 7 x 3
    Sigma: Vector of size 7
    xi: Vector of size 7
    """
    # Extract position, velocity, mass from state vector, as 2D column vectors
    r = np.vstack(x[0:3])
    rz = x[2]
    distance = np.linalg.norm(r)
    v = np.vstack(x[3:6])
    speed = np.linalg.norm(v)
    m = x[6]
    T = np.vstack(u)
    # Get atmospheric densities
    rho = rho_func(x[0:3])
    drho = drho_func(x[0:3])
    # Compute partial derivative terms according to appendix C.2
    # Partial derivative of a_g with respect to position
    Dr_ag = (-const.MU/(distance**3)*np.eye(3)
             + 3*const.MU/(distance**5)*np.dot(r, r.T))
    # Partial derivative of a_J2 with repspect to position
    if use_J2:
        kJ2 = 1.5*const.J2*const.MU*const.R_E
        rz_norm = rz/distance
        GJ2 = np.diag([5*(rz_norm**2)-1, 5*(rz_norm**2)-1, 5*(rz_norm**2)-3])
        ddr = (5*(rz**2)*(-2*(r.T/(distance**4)))
                + (5/(distance**2))*np.array([[0, 0, 2*rz]]))
        Dr_aJ2 = (np.matmul(kJ2*GJ2*r, -5*r.T/(distance**7))
                + kJ2/(distance**5) *
                np.vstack([r[0, 0]*ddr, r[1, 0]*ddr, rz*ddr])
                + kJ2/(distance**5) * GJ2 * np.eye(3))
    else:
        Dr_aJ2 = np.zeros((3,3))
    # Partial derivative of a_D with respect to position, velocity, mass
    if use_drag:
        Dr_aD = np.matmul(-(const.CD*const.S/(2*m))*speed*v,
                        drho * r.T/distance)
        Dv_aD = -((rho*const.CD*const.S)/(2*m))*(speed*np.eye(3)
                                                + (1/(speed))*np.matmul(v, v.T))
        Dm_aD = ((rho*const.CD*const.S)/(2*m**2))*speed*v
    else:
        Dr_aD = np.zeros((3,3))
        Dv_aD = np.zeros((3,3))
        Dm_aD = np.zeros((3,3))
    # Partial derivative of a_T with respect to mass, thrust
    Dm_aT = -T/m
    DT_aT = (1/m)*np.eye(3)

    # Build Dxf
    Dxf = np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))]),
                     np.hstack([Dr_ag + Dr_aJ2 + Dr_aD, Dv_aD, Dm_aD + Dm_aT]),
                     np.hstack([np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 1))])])

    # Build Duf
    DT_fm = -(T.T)/(const.G0*const.ISP*np.linalg.norm(T))
    Duf = np.vstack([np.zeros((3, 3)), DT_aT, DT_fm])

    # Calculate and output A, B, Sigma, xi
    A = tf*Dxf
    B = tf*Duf
    Sigma = f(x, u)
    xi = -(np.dot(A, x) + np.dot(B, u))
    return A, B, Sigma, xi


def discretize(x, u, t):
    Ak = []
    Bk = []


    return Ak, Bk
