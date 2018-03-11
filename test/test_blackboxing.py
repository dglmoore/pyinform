# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np
from pyinform.utils import black_box
from pyinform.error import InformError

class TestBlackBoxing(unittest.TestCase):
    def test_empty_series(self):
        with self.assertRaises(ValueError):
            black_box([])

    def test_negative_state(self):
        with self.assertRaises(InformError):
            black_box([[0, -1, 1], [1, 0, 1]])

        with self.assertRaises(InformError):
            black_box([[0, 1, 1], [1, 0, -1]])

    def test_invalid_state(self):
        with self.assertRaises(InformError):
            black_box([[0, 2, 1], [1, 0, 1]])

        with self.assertRaises(InformError):
            black_box([[0, 1, 1], [1, 0, 2]])

    def test_invalid_history(self):
        series = [[0,1,1], [1,0,1]]

        with self.assertRaises(InformError):
            black_box(series, k=(0,0))

        with self.assertRaises(InformError):
            black_box(series, k=(0,1))

        with self.assertRaises(InformError):
            black_box(series, k=(1,0))

        with self.assertRaises(InformError):
            black_box(series, k=(-1,1))

        with self.assertRaises(InformError):
            black_box(series, k=(-1,-1))

    def test_invalid_future(self):
        series = [[0,1,1], [1,0,1]]

        with self.assertRaises(InformError):
            black_box(series, l=(-1,0))

        with self.assertRaises(InformError):
            black_box(series, l=(0,-1))

        with self.assertRaises(InformError):
            black_box(series, l=(-1,-1))

    def test_long_history_future(self):
        series = [[0,1,1], [1,0,1]]
        with self.assertRaises(InformError):
            black_box(series, k=(4,1))

        with self.assertRaises(InformError):
            black_box(series, k=(1,4))

        with self.assertRaises(InformError):
            black_box(series, k=(0,1), l=(4,0))

        with self.assertRaises(InformError):
            black_box(series, k=(0,1), l=(1,3))

    def test_encoding_error(self):
        series = [
            0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
            0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1
        ]
        with self.assertRaises(InformError):
            black_box(series, k=31)
        with self.assertRaises(InformError):
            black_box(series, b=4, k=16)
        with self.assertRaises(InformError):
            black_box(series, k=29, l=2)

        series = [
            [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0],
            [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0],
        ]
        with self.assertRaises(InformError):
            black_box(series, k=(15,16))
        with self.assertRaises(InformError):
            black_box(series, b=(2,4), k=(15,8))
        with self.assertRaises(InformError):
            black_box(series, b=(2,3), k=(10,5), l=(4,6))

    def test_single_series(self):
        series = [0,1,1,0,1,1,0,0]

        print(black_box(series))
        self.assertTrue(
                np.array_equal(series, black_box(series)))
        self.assertTrue(
                np.array_equal(series, black_box(series, k=1)))
        self.assertTrue(
                np.array_equal(series, black_box(series, k=1, l=0)))
        self.assertTrue(
                np.array_equal([1,3,2,1,3,2,0], black_box(series, k=2)))
        self.assertTrue(
                np.array_equal([1,3,2,1,3,2,0], black_box(series, k=1, l=1)))
        self.assertTrue(
                np.array_equal([1,3,2,1,3,2,0], black_box(series, l=1)))
        self.assertTrue(
                np.array_equal([3,6,5,3,6,4], black_box(series, k=3)))
        self.assertTrue(
                np.array_equal([3,6,5,3,6,4], black_box(series, k=1, l=2)))
        self.assertTrue(
                np.array_equal([3,6,5,3,6,4], black_box(series, l=2)))

        series = [0,1,2,0,1,1,0,2]
        self.assertTrue(
                np.array_equal([5,15,19,4,12,11], black_box(series, k=1, l=2)))
        self.assertTrue(
                np.array_equal([5,15,19,4,12,11], black_box(series, l=2)))

    def test_single_series_ensemble(self):
        series = [
            [0,1,1,0,1,1,0,0],
            [0,0,1,1,0,1,0,1],
        ]
        self.assertEqual(series, black_box(series))
        self.assertEqual(series, black_box(series, k=1))
        self.assertEqual([
            [1,3,2,1,3,2,0],
            [0,1,3,2,1,2,1],
        ], black_box(series, k=2))
        self.assertEqual([
            [1,3,2,1,3,2,0],
            [0,1,3,2,1,2,1],
        ], black_box(series, k=1, l=1))
        self.assertEqual([
            [1,3,2,1,3,2,0],
            [0,1,3,2,1,2,1],
        ], black_box(series, l=1))
        self.assertEqual([
            [3,6,5,3,6,4],
            [1,3,6,5,2,5],
        ], black_box(series, k=3))
        self.assertEqual([
            [3,6,5,3,6,4],
            [1,3,6,5,2,5],
        ], black_box(series, k=2, l=1))
        self.assertEqual([
            [3,6,5,3,6,4],
            [1,3,6,5,2,5],
        ]), black_box(series, k=1, l=2)
        self.assertEqual([
            [3,6,5,3,6,4],
            [1,3,6,5,2,5],
        ], black_box(series, l=2))

        series = [
            [0,1,2,0,1,1,0,2],
            [2,1,1,2,0,0,1,2],
        ]
        self.assertEqual([
            [ 5,15,19, 4,12,11],
            [22,14,15,18, 1, 5],
        ], black_box(series, k=1, l=2))

    def test_multiple_series(self):
        series = [
            [0,1,1,0,1,1,0,0],
            [0,0,1,1,0,1,0,1],
        ]
        self.assertEqual([0,2,3,1,2,3,0,1], black_box(series))
        self.assertEqual([0,2,3,1,2,3,0,1], black_box(series, k=(1,1)))
        self.assertEqual([2,7,5,2,7,4,1], black_box(series, k=(2,1)))
        self.assertEqual([4,5,3,6,5,2,1], black_box(series, k=(1,2)))
        self.assertEqual([4,13,11,6,13,10,1], black_box(series, k=(2,2)))
        self.assertEqual([2,6,5,3,6,5,0], black_box(series, l=(1,0)))
        self.assertEqual([2,6,5,3,6,5,0], black_box(series, k=(1,1), l=(1,0)))
        self.assertEqual([0,5,7,2,5,6,1], black_box(series, l=(0,1)))
        self.assertEqual([0,5,7,2,5,6,1], black_box(series, k=(1,1), l=(0,1)))
        self.assertEqual([4,13,11,6,13,10,1], black_box(series, l=(1,1)))
        self.assertEqual([4,13,11,6,13,10,1],
                black_box(series, k=(1,1), l=(1,1)))
        self.assertEqual([6,13,11,6,13,8], black_box(series, k=(2,1), l=(1,0)))

        series = [
            [0,1,2,0,1,1,0,2],
            [0,0,1,1,0,1,0,1],
        ]
        self.assertEqual([10,31,39,8,25,22],
                black_box(series, k=(2,1), l=(1,0)))

        series = [
            [0,1,1,0,1,1,0,0],
            [0,2,1,1,0,2,0,1],
        ]
        self.assertEqual([11,19,16,9,20,12],
                black_box(series, k=(2,1), l=(1,0)))

        series = [
            [0,3,1,0,1,1,2,0],
            [0,1,1,1,0,1,0,1],
        ]
        self.assertEqual([27,105,35,10,45,48],
                black_box(series, k=(2,1), l=(1,0)))

        series = [
            [0,1,1,1,0,1,0,1],
            [0,3,1,0,1,1,2,0],
        ]
        self.assertEqual([15,29,24,21,9,22],
                black_box(series, k=(2,1), l=(1,0)))

    def test_multiple_series_ensemble(self):
        series = [
            [
                [0,1,1,0,1,1,0,0],
                [0,0,1,1,0,1,0,1],
            ],
            [
                [1,1,0,1,0,0,1,0],
                [0,0,0,1,0,0,1,0],
            ],
        ]
        self.assertEqual([
            [1,3,2,1,2,2,1,0],
            [0,0,2,3,0,2,1,2],
        ], black_box(series))
        self.assertEqual([
            [1,3,2,1,2,2,1,0],
            [0,0,2,3,0,2,1,2],
        ], black_box(series, k=(1,1)))
        self.assertEqual([
            [3,6,5,2,6,5,0],
            [0,2,7,4,2,5,2],
        ], black_box(series, k=(2,1)))
        self.assertEqual([
            [7,6,1,6,4,1,2],
            [0,4,5,2,4,1,6],
        ], black_box(series, k=(1,2)))
        self.assertEqual([
            [7,14,9,6,12,9,2],
            [0,4,13,10,4,9,6],
        ], black_box(series, k=(2,2)))
        self.assertEqual([
            [3,7,4,3,6,4,1],
            [0,2,6,5,2,4,3],
        ], black_box(series, l=(1,0)))
        self.assertEqual([
            [3,7,4,3,6,4,1],
            [0,2,6,5,2,4,3],
        ], black_box(series, k=(1,1), l=(1,0)))
        self.assertEqual([
            [3,6,5,2,4,5,2],
            [0,0,5,6,0,5,2],
        ], black_box(series, l=(0,1)))
        self.assertEqual([
            [3,6,5,2,4,5,2],
            [0,0,5,6,0,5,2],
        ], black_box(series, k=(1,1), l=(0,1)))
        self.assertEqual([
            [7,14,9,6,12,9,2],
            [0,4,13,10,4,9,6],
        ], black_box(series, l=(1,1)))
        self.assertEqual([
            [7,14,9,6,12,9,2],
            [0,4,13,10,4,9,6],
        ], black_box(series, k=(1,1), l=(1,1)))
        self.assertEqual([
            [7,12,11,6,12,9],
            [2,6,13,10,4,11],
        ], black_box(series, k=(2,1), l=(1,0)))


        series = [
            [
                [0,1,2,0,1,1,0,2],
                [0,0,1,1,0,1,0,1],
            ],
            [
                [1,0,1,1,0,1,0,0],
                [1,1,0,1,0,0,1,0],
            ],
        ]
        self.assertEqual([
            [10,31,39, 8,25,22],
            [ 3, 8,25,20, 6,21],
        ], black_box(series, k=(2,1), l=(1,0)))

    
        series = [
            [
                [1,0,1,1,0,1,0,0],
                [1,1,0,1,0,0,1,0],
            ],
            [
                [0,1,2,0,1,1,0,2],
                [0,0,1,1,0,1,0,1],
            ],
        ]
        self.assertEqual([
            [16,11,18,16,7,12],
            [18,16, 7,12,4, 6],
        ], black_box(series, k=(2,1), l=(1,0)))

        series = [
            [
                [0,3,1,0,1,1,2,0],
                [0,1,2,1,0,1,3,1],
            ],
            [
                [0,1,1,0,1,0,1,1],
                [0,0,1,1,0,1,0,0],
            ],
        ]
        self.assertEqual([
            [27,105,34,11,44,49],
            [12, 51,73,34,15,58],
        ], black_box(series, k=(2,1), l=(1,0)))

        series = [
            [
                [0,1,1,0,1,0,1,1],
                [0,0,1,1,0,1,0,0],
            ],
            [
                [0,3,1,0,1,1,2,0],
                [0,1,2,1,0,1,3,1],
            ],
        ]
        self.assertEqual([
            [15,25,20, 9,21,14],
            [ 5,14,25,20, 9,19],
        ], black_box(series, k=(2,1), l=(1,0)))
