# Test for Satellite class
import unittest
from satellite import Satellite
import numpy as np

class TestSatellite(unittest.TestCase):

    def test_init(self):
        r = np.array([1.0, 2.5, 3])
        v = np.array([0, 0, 1.0])
        m = 1000
        t = np.array([0, 0, 1500.0])
        s = Satellite(position=r, velocity=v, mass=m, thrust=t)
        print(str(s))
        self.assertTrue(all(s.position == r))
        self.assertTrue(all(s.velocity == v))
        self.assertTrue(all(s.thrust == t))
        self.assertEqual(s.mass, m)

if __name__ == '__main__':
    unittest.main()