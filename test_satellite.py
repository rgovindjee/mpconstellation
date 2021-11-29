# Test for Satellite class
import unittest
from satellite import Satellite
import numpy as np

class TestSatellite(unittest.TestCase):

    def test_init(self):
        """
        Test basic initialization of satellite
        """
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

    def test_uuid(self):
        """
        Create 10000 satellites and make sure their UUIDs are all unique
        """
        sats = [Satellite() for x in range(10000)]
        sat_ids = [s.id for s in sats]
        sat_id_set = set(sat_ids)
        self.assertTrue(len(sat_ids) == len(sat_id_set))

if __name__ == '__main__':
    unittest.main()
