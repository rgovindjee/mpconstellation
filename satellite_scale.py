import numpy as np
from constants import *

class SatelliteScale:
    """
    A SatelliteScale object contains methods to normalize and redimensionalize
    states and thrusts with a defined set of scaling factors.
    """
    def __init__(self, x = None, sat = None):
        """Initializes a SatelliteScale object.

        Initalizes a SatelliteScale object by scaling factors determined from
        an input state x.

        Args:
            x: a 7 vector of states [position, velocity, mass]
            sat: a satellite object. If a satellite object is pass in, inputs
                 for keyword 'x' are ignored.

        Returns:
            a SatelliteState object with methods to normalize/redimensionalize.
        """
        if sat is not None:
            x = sat.get_state_vector()
        elif x is None: # If no inputs, set scale factor to 1
            x = np.array([1,0,0,0,0,0,1])
        # Designer units (pg. 20)
        self._r0  = np.linalg.norm(x[0:3])
        self._s0 = 2*np.pi*np.sqrt(self._r0**3/MU_EARTH)
        self._v0 = self._r0/self._s0
        self._a0 = self._r0/self._s0**2
        self._m0 = x[6]
        self._T0 = self._m0*self._r0/self._s0**2
        self._mu0 = self._r0**3/self._s0**2

    def get_normalized_constants(self):
        """Get global constants normalized by the scaling factor

        Returns:
            a Constants object normalized by the scaling factor
        """
        return Constants(MU=MU_EARTH/self._mu0, R_E=R_EARTH/self._r0, J2=J2,
                         G0=G0/self._a0, ISP=ISP/self._s0, S=S/self._r0**2,
                         R0=self._r0, RHO=self._m0/self._r0**3)

    def redim_state(self, x):
        """Redimensionalizes state vector(s) x by the scaling factor

        Args:
            x: a 7 vector or 7 x N array of nondimensional states
               [position, velocity, mass]

        Returns:
            dimensionalized states, same shape as x
        """
        if x.ndim == 1:
            return np.concatenate([x[0:3]*self._r0, x[3:6]*self._v0, [x[6]*self._m0]])
        else:
            assert x.shape[0] == 7, "If x is 2D, must be shaped as 7 x N"
            return np.vstack([x[0:3,:]*self._r0, x[3:6,:]*self._v0, x[6,:]*self._m0])

    def normalize_state(self, x):
        """Normalizes state vector(s) x by the scaling factor

        Args:
            x: a 7 vector or 7 x N array of dimensional states
               [position, velocity, mass]

        Returns:
            normalized states, same shape as x
        """
        if x.ndim == 1:
            return np.concatenate([x[0:3]/self._r0,
                                   x[3:6]/self._v0,
                                   np.array([x[6]/self._m0])])
        else:
            assert x.shape[0] == 7, "If x is 2D, must be shaped as 7 x N"
            return np.vstack([x[0:3,:]/self._r0, x[3:6,:]/self._v0, x[6,:]/self._m0])

    def redim_thrust(self, u):
        """Redimensionalizes thrust vector(s) u by the scaling factor

        Args:
            u: a 3 vector or 3 x N array of dimensional thrust

        Returns:
            normalized thrust, same shape as u
        """
        return u * self._T0

    def normalize_thrust(self, u):
        """Normalizes thrust vector(s) u by the scaling factor

        Args:
            u: a 3 vector or 3 x N array of dimensional thrust

        Returns:
            normalized thrust, same shape as u
        """
        return u / self._T0
