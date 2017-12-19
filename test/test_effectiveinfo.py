# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.error import InformError
from pyinform.effectiveinfo import *

class TestEffectiveInfo(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            effective_info([])
        with self.assertRaises(ValueError):
            effective_info([[]])

    def test_not_matrix(self):
        with self.assertRaises(ValueError):
            effective_info(5)

        with self.assertRaises(ValueError):
            effective_info([0,1])

    def test_not_square(self):
        with self.assertRaises(ValueError):
            effective_info([5])

        with self.assertRaises(ValueError):
            effective_info([[1,0]])

        with self.assertRaises(ValueError):
            effective_info([[0.5, 0.5], [1]])

    def test_zero_row(self):
        with self.assertRaises(InformError):
            tpm = np.asarray([[0, 1], [0, 0]], dtype=np.float64)
            effective_info(tpm)

    def test_not_normalized(self):
        with self.assertRaises(InformError):
            effective_info([[0.5, 0.5], [0.5, 0.25]])

        with self.assertRaises(InformError):
            effective_info([[0.5, 0.5], [0.5, 0.75]])

    def test_intervention_not_vector(self):
        with self.assertRaises(ValueError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], 0)
        with self.assertRaises(ValueError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], [[0],[1]])

    def test_intervention_size_mismatch(self):
        with self.assertRaises(ValueError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], [1])

        with self.assertRaises(ValueError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], [0,0.5,0.5])

    def test_intervention_zero(self):
        with self.assertRaises(InformError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], [0, 0])

    def test_intervention_negative(self):
        with self.assertRaises(InformError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], [0.8, -0.2])

    def test_intervention_not_normalized(self):
        with self.assertRaises(InformError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], [0.5, 0.25])

        with self.assertRaises(InformError):
            effective_info([[0.5, 0.5], [0.25, 0.75]], [0.5, 0.75])

    def test_uniform_intervention(self):
        tpm = [[0.2, 0.8], [0.75, 0.25]]
        self.assertAlmostEqual(0.231593, effective_info(tpm), places=6)

        tpm = [[1.0/3, 1.0/3, 1.0/3],
               [0.250, 0.750, 0.000],
               [0.125, 0.500, 0.375]]
        self.assertAlmostEqual(0.202701, effective_info(tpm), places=6)

    def test_nonuniform_intervention(self):
        tpm = [[0.2, 0.8], [0.75, 0.25]]
        self.assertAlmostEqual(0.174227, effective_info(tpm, [0.25, 0.75]),
                places=6)

        tpm = [[1.0/3, 1.0/3, 1.0/3],
               [0.250, 0.750, 0.000],
               [0.125, 0.500, 0.375]]
        self.assertAlmostEqual(0.172498, effective_info(tpm, [0.3, 0.25, 0.45]),
                places=6)

    def test_from_hoel(self):
        # E. Hoel, "When the map is better than the territory", arXiv:1612.09592
        tpm = [[0,0,1,0],
               [1,0,0,0],
               [0,0,0,1],
               [0,1,0,0]]
        self.assertAlmostEqual(2.0, effective_info(tpm))

        tpm = [[1.0/3, 1.0/3, 1.0/3, 0.000],
               [1.0/3, 1.0/3, 1.0/3, 0.000],
               [0.000, 0.000, 0.000, 1.000],
               [0.000, 0.000, 0.000, 1.000]]
        self.assertAlmostEqual(1.0, effective_info(tpm))

        tpm = [[1.0/4, 1.0/4, 1.0/4, 1.0/4],
               [1.0/4, 1.0/4, 1.0/4, 1.0/4],
               [1.0/4, 1.0/4, 1.0/4, 1.0/4],
               [1.0/4, 1.0/4, 1.0/4, 1.0/4]]
        self.assertAlmostEqual(0.0, effective_info(tpm))

        tpm = [[1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]]
        self.assertAlmostEqual(0.543564, effective_info(tpm), places=6)

        tpm = [[1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5, 0.000, 0.000, 0.000],
               [1.0/7, 3.0/7, 1.0/7, 0.000, 1.0/7, 0.000, 1.0/7, 0.000],
               [0.000, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 0.000],
               [1.0/7, 0.000, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 2.0/7, 0.000],
               [1.0/9, 2.0/9, 2.0/9, 1.0/9, 0.000, 2.0/9, 1.0/9, 0.000],
               [1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7, 0.000],
               [1.0/6, 1.0/6, 0.000, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 0.000],
               [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]]
        self.assertAlmostEqual(0.805890, effective_info(tpm), places=6)

        tpm = [[1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8],
               [1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8],
               [1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8],
               [1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8],
               [1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8],
               [1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8],
               [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000],
               [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]]
        self.assertAlmostEqual(0.630241, effective_info(tpm), places=6)

if __name__ == "__main__":
    unittest.main()
