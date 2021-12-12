import numpy as np
import uuid

class Satellite:
    """
    A satellite object with relevant state variables: position, velocity, mass,
    and thrust
    """
    def __init__(self, position=np.array([0., 0., 0.]),
                       velocity=np.array([0., 0., 0.]),
                       mass=0.):
        """
        Arguments:
            position: 3 x 1 array (m)
            velocity: 3 x 1 array (m/s)
            mass: scalar (kg)
        The position and velocity vectors are coordinated in an inertial
        reference frame placed at the center of the central body with its
        z-axis aligned with rotation axis of the central body
        """
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.id = uuid.uuid4().int  # Assign a unique numeric identifier to the satellite

    def get_state_vector(self):
        """
        Returns:
            The (7,) state vector of the satellite
        """
        return np.concatenate([self.position, self.velocity, np.array([self.mass])])

    def update_state_vector(self, state):
        """
        Takes in a (7, ) state vector and updates the class properties accordingly
        """
        self.position = state[0:3]
        self.velocity = state[3:6]
        self.mass = state[6]

    def __str__(self):
        return (
            f'Satellite {hex(self.id)} with mass {self.mass}:\n'
            f'position: {self.position}\n'
            f'velocity: {self.velocity}'
        )
