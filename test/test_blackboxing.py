# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.utils import black_box
from pyinform.error import InformError

class TestBlackBoxing(unittest.TestCase):
    def test_null_series(self):
        with self.assertRaises(ValueError):
            black_box([])
