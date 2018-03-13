# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np
from pyinform.utils import partitionings
from pyinform.error import InformError

class TestBlackBoxing(unittest.TestCase):
    def test_partition_negative(self):
        with self.assertRaises(ValueError):
            list(partitionings(-1))

    def test_partition_zero(self):
        with self.assertRaises(ValueError):
            list(partitionings(0))

    def test_parition_one(self):
        self.assertTrue(np.array_equal([[0]], list(partitionings(1))))

    def test_parition_two(self):
        self.assertTrue(np.array_equal([[0,0], [0,1]], list(partitionings(2))))

    def test_parition_three(self):
        expect = [
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [0,1,2],
        ]
        self.assertTrue(np.array_equal(expect, list(partitionings(3))))

    def test_parition_four(self):
        expect = [
            [0,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [0,0,1,1],
            [0,0,1,2],
            [0,1,0,0],
            [0,1,0,1],
            [0,1,0,2],
            [0,1,1,0],
            [0,1,1,1],
            [0,1,1,2],
            [0,1,2,0],
            [0,1,2,1],
            [0,1,2,2],
            [0,1,2,3],
        ]
        self.assertTrue(np.array_equal(expect, list(partitionings(4))))

    def test_bell_numbers(self):
        bell_numbers = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
        for i, b in enumerate(bell_numbers):
            self.assertEqual(b, len(list(partitionings(i+1))))
