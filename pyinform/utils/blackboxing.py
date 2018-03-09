# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def black_box(series):
    pass

_inform_black_box = _inform.inform_black_box
_inform_black_box.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_ulong, POINTER(c_int), POINTER(c_ulong), POINTER(c_ulong), POINTER(c_int), POINTER(c_int)]
_inform_black_box.restype = POINTER(c_int)
