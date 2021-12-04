import numpy as np
import uuid

class Satellite:
    """
    A satellite object with relevant state variables: position, velocity, mass,
    and thrust
    """
    def __init__(self, position=np.array([0., 0., 0.]),
                       velocity=np.array([0., 0., 0.]),
                       mass=0.,
                       thrust=np.array([0., 0., 0.])):
        """
        Arguments:
            position: 3 x 1 array (m)
            velocity: 3 x 1 array (m/s)
            mass: scalar (kg)
            thrust: 3 x 1 array (N)
        The position and velocity vectors are coordinated in an inertial
        reference frame placed at the center of the central body with its
        z-axis aligned with rotation axis of the central body
        """
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.thrust = thrust
        self.id = uuid.uuid4().int  # Assign a unique numeric identifier to the satellite

    def __str__(self):
        return (
            f'Satellite {hex(self.id)} with mass {self.mass}:\n'
            f'position: {self.position}\n'
            f'velocity: {self.velocity}'
        )
