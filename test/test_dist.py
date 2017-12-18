# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import unittest

from pyinform.dist import Dist

class TestDist(unittest.TestCase):

    def test_alloc_negative(self):
        with self.assertRaises(ValueError):
            Dist(-1)

    def test_alloc_zero(self):
        with self.assertRaises(ValueError):
            Dist(0)

    def test_alloc_size(self):
        d = Dist(5)
        self.assertEqual(5, d.__len__())
        self.assertEqual(5, len(d))

    def test_from_hist_empty(self):
        with self.assertRaises(ValueError):
            Dist.from_hist([])
        
        with self.assertRaises(ValueError):
            Dist.from_hist(np.array([]))

    def test_from_hist_multidimensional(self):
        with self.assertRaises(ValueError):
            Dist.from_hist([[1,1,2,2]])

        with self.assertRaises(ValueError):
            Dist.from_hist(np.array([[1,1,2,2]]))

        with self.assertRaises(ValueError):
            Dist.from_hist([[1,2,3],[1,2]])

        with self.assertRaises(ValueError):
            Dist.from_hist(np.asarray([[1,2,3],[1,2]]))

    def test_from_hist_list(self):
        lst = [1,1,2,2]
        d = Dist.from_hist(lst)
        self.assertEqual(4, len(d))
        for i in range(len(d)):
            self.assertEqual(lst[i], d[i])

    def test_from_hist_list_copies(self):
        lst = [0,0,0,0]
        d = Dist.from_hist(lst)
        for i in range(len(d)):
            d[i] = i
            self.assertEqual(d[i], i)
            self.assertEqual(lst[i], 0)

    def test_from_hist_array(self):
        arr = np.array([1,1,2,2], dtype=np.uint32)
        d = Dist.from_hist(arr)
        self.assertEqual(4, len(d))
        for i in range(len(arr)):
            self.assertEqual(arr[i], d[i])

    def test_from_hist_array_copies(self):
        arr = np.array([0,0,0,0], dtype=np.uint32)
        d = Dist.from_hist(arr)
        for i in range(len(d)):
            d[i] = i
            self.assertEqual(d[i], i)
            self.assertEqual(arr[i], 0)

    def test_from_probs_empty(self):
        with self.assertRaises(ValueError):
            Dist.from_probs([])

    def test_from_probs_negative(self):
        with self.assertRaises(ValueError):
            Dist.from_probs(-1)

        with self.assertRaises(ValueError):
            Dist.from_probs([-0.5, 0.5])

    def test_from_probs_unity(self):
        with self.assertRaises(ValueError):
            Dist.from_probs(0.9)

        with self.assertRaises(ValueError):
            Dist.from_probs(1.0 - 1e-6)

        with self.assertRaises(ValueError):
            Dist.from_probs([0.5, 0.4])

        with self.assertRaises(ValueError):
            Dist.from_probs([0.5-1e-9, 0.5-1e-9])

        Dist.from_probs(1.0 - 1e-9)
        Dist.from_probs(1.0 - 1e-7, tol=1e-6)

    def test_from_data_empty(self):
        with self.assertRaises(ValueError):
            Dist.from_data([])

    def test_from_data(self):
        d = Dist.from_data([0,0,1,1,0,1,0])
        self.assertEqual([4,3], d[:])

        d = Dist.from_data([0,1,1,2,1,0])
        self.assertEqual([2,3,1], d[:])

        d = Dist.from_data([2,2,2], n=4)
        self.assertEqual([0,0,3,0], d[:])

        d = Dist.from_data([0,1,2], n=2)
        self.assertEqual([1,1,1], d[:])

    def test_uniform_zero(self):
        with self.assertRaises(ValueError):
            Dist.uniform(0)

    def test_uniform_negative(self):
        with self.assertRaises(ValueError):
            Dist.uniform(-1)

    def test_uniform(self):
        self.assertEqual([1],       Dist.uniform(1)[:])
        self.assertEqual([1,1],     Dist.uniform(2)[:])
        self.assertEqual([1,1,1],   Dist.uniform(3)[:])
        self.assertEqual([1,1,1,1], Dist.uniform(4)[:])

    def test_resize_negative(self):
        d = Dist(3)
        with self.assertRaises(ValueError):
            d.resize(-1)

    def test_resize_zero(self):
        d = Dist(3)
        with self.assertRaises(ValueError):
            d.resize(0)

    def test_resize_grow(self):
        d = Dist(3)
        for i in range(len(d)):
            d[i] = i+1
        self.assertEqual(3, len(d))
        self.assertEqual(6, d.counts)

        d.resize(5)
        self.assertEqual(5, len(d))
        self.assertEqual(6, d.counts)
        for i in range(3):
            self.assertEqual(i+1, d[i])
        for i in range(3,len(d)):
            self.assertEqual(0, d[i])

    def test_resize_shrink(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i+1
        self.assertEqual(5, len(d))
        self.assertEqual(15, d.counts)

        d.resize(3)
        self.assertEqual(3, len(d))
        self.assertEqual(6, d.counts)
        for i in range(len(d)):
            self.assertEqual(i+1, d[i])

    def test_copy(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i+1
        self.assertEqual(5, len(d))
        self.assertEqual(15, d.counts)

        e = d.copy()
        self.assertEqual(5, len(d))
        self.assertEqual(15, d.counts)
        for i in range(len(d)):
            self.assertEqual(e[i], d[i])

        d[0] = 5
        self.assertNotEqual(e[0], d[0])

    def test_get_bounds_error(self):
        d = Dist(2)
        with self.assertRaises(IndexError):
            d[-1]

        with self.assertRaises(IndexError):
            d[3]

    def test_set_bounds_error(self):
        d = Dist(2)
        with self.assertRaises(IndexError):
            d[-1] = 3

        with self.assertRaises(IndexError):
            d[3] = 1

    def test_set_negative(self):
        d = Dist(2)
        d[0] = 5
        self.assertEqual(5, d[0])

        d[0] = -1
        self.assertEqual(0, d[0])

    def test_get_and_set(self):
        d = Dist(2)
        self.assertEqual(0, d[0])

        d[0] = 4
        self.assertEqual(4, d[0])
        self.assertEqual(0, d[1])

        d[1] = 2
        self.assertEqual(2, d[1])
        self.assertEqual(4, d[0])

    def test_get_slice(self):
        d = Dist(5)
        self.assertEqual([0,0,0,0,0], d[:])
        self.assertEqual([0,0,0], d[:3])
        self.assertEqual([0,0], d[3:])

        d = Dist.from_hist([0,1,2,3,4])
        self.assertEqual([0,1,2,3,4], d[:])
        self.assertEqual([4,3,2,1,0], d[::-1])
        self.assertEqual([1,2,3], d[1:4])

    def test_counts(self):
        d = Dist(2)
        self.assertEqual(0, d.counts)

        d[0] = 3
        self.assertEqual(3, d.counts)

        d[0] = 2
        self.assertEqual(2, d.counts)

        d[1] = 3
        self.assertEqual(5, d.counts)

        d[0] = 0
        d[1] = 0
        self.assertEqual(0, d.counts)

    def test_valid(self):
        d = Dist(2)
        self.assertFalse(d.is_valid)
        d[0] = 2
        self.assertTrue(d.is_valid)
        d[1] = 2
        self.assertTrue(d.is_valid)
        d[0] = 0
        self.assertTrue(d.is_valid)
        d[1] = 0
        self.assertFalse(d.is_valid)

    def test_tick_bounds_error(self):
        d = Dist(2)
        with self.assertRaises(IndexError):
            d.tick(-1)

        with self.assertRaises(IndexError):
            d.tick(3)

    def test_tick(self):
        d = Dist(2)

        self.assertEqual(1, d.tick(0))
        self.assertEqual(2, d.tick(0))
        self.assertEqual(2, d.counts)
        self.assertTrue(d.is_valid)

    def test_accumulate_invalid(self):
        d = Dist(3)

        with self.assertRaises(IndexError):
            d.accumulate(-1)

        with self.assertRaises(IndexError):
            d.accumulate(3)

        self.assertEqual([0,0,0], d[:])

        with self.assertRaises(IndexError):
            d.accumulate([0,1,2,3])
        self.assertEqual([1,1,1], d[:])

        with self.assertRaises(IndexError):
            d.accumulate([0,1,2,-1])
        self.assertEqual([2,2,2], d[:])

    def test_accumulate(self):
        d = Dist(3)
        self.assertEqual(0, d.accumulate([]))
        self.assertEqual(3, d.accumulate([0,1,2]))
        self.assertEqual([1,1,1], d[:])

        self.assertEqual(4, d.accumulate([1,1,0,1]))
        self.assertEqual([2,4,1], d[:])

    def test_probability_invalid(self):
        d = Dist(5)
        for i in range(len(d)):
            with self.assertRaises(ValueError):
                d.probability(i)

    def test_probabilify_bounds_error(self):
        d = Dist(2)
        d[0] = 1
        with self.assertRaises(IndexError):
            d.probability(-1)

        with self.assertRaises(IndexError):
            d.probability(3)

    def test_probability(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i+1
        for i in range(len(d)):
            self.assertAlmostEqual((i+1)/15., d.probability(i))

    def test_dump_invalid(self):
        d = Dist(2)
        with self.assertRaises(ValueError):
            d.dump()

    def test_dump(self):
        d = Dist(5)
        for i in range(1, len(d)):
            d[i] = i+1
        self.assertEqual(14, d.counts)
        probs = d.dump()
        self.assertTrue((probs == np.array([0., 2./14, 3./14, 4./14, 5./14])).all())

if __name__ == "__main__":
    unittest.main()
