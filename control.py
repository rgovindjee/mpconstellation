import numpy as np

class Controller:
    """
    Controllers track one or more satellites and produce thrust commands for each satellite under control.
    """
    def __init__(self, sats=[]):
        """
        Arguments:
            sats: list of Satellite objects
        """
        self.sats = sats
        self.sat_ids = set(s.id for s in sats)

    def get_u_func(sat_id=None):
        """
        Arguments:
            sat_id: id of satellite to return output for.
        Returns an open-loop control function u(t) for a given satellite.
        """
        #Satellite id irrelevant for base controller.
        zero_thrust = np.array([0., 0., 0.])
        u = lambda x, tau: zero_thrust
        return u

class ConstantThrustController(Controller):
    def __init__(self, sats=[], thrust=np.array([1., 1., 1.])):
        """
        Arguments:
            sats: list of Satellite objects
            thrust: 3-element array of thrust in x, y, z direction
        """
        super().__init__(sats)
        self.thrust = thrust

    def get_u_func(self, sat_id=None):
        """
        Arguments:
            sat_id: id of satellite to return output for.
        """
        u = lambda x, tau: self.thrust
        return u

class ConstantTangentialThrustController(Controller):
    def __init__(self, sats=[], tangential_thrust=1):
        """
        Arguments:
            sats: list of Satellite objects
            tangential_thrust: scalar, magnitude of tangential thrust
        """
        super().__init__(sats)
        self.tangential_thrust = tangential_thrust
    

    def compute_rotation(self, x):
        """
        Arguments:
            x: State vector of satellite
        Returns:
            3x3 rotation matrix from the RTN frame to ECI frame
        """
        r = x[0:3]
        v = x[3:6]
        r_hat = r/np.linalg.norm(r)
        h = np.cross(r, v)
        h_hat = h/np.linalg.norm(h)
        t_hat = np.cross(h_hat, r_hat)
        return np.column_stack([r_hat, t_hat, h_hat])
    
    
    def get_u_func(self, sat_id=None):
        u = lambda x, tau: self.compute_rotation(x) @ np.array([0,self.tangential_thrust,0])
        return u


class OptimalController(Controller):
    def __init__(self, sats=[], objective=None):
        """
        Arguments:
            sats: list of Satellite objects
            thrust: 3-element array of thrust in x, y, z direction
            objective: desired final arrangement of satellites?
        """
        super().__init__(sats)
    # TODO(rgg, cm): flesh this out with optimization formulation
