# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_double, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def to_tpm(series, b=0):
    """
    Construct a transition probability matrix from a time series.

        >>> to_tpm([0,0,1,1,0,1,0])
        array([[ 0.33333333,  0.66666667],
               [ 0.66666667,  0.33333333]])
        >>> to_tpm([[0,2,1,1], [0,1,1,1], [1,0,2,0]])
        array([[ 0.        ,  0.33333333,  0.66666667],
               [ 0.25      ,  0.75      ,  0.        ],
               [ 0.5       ,  0.5       ,  0.        ]])

    :param series: the time series
    :param b: the base of the time series (optional)
    :return: the transition probability matrix
    """
    xs = np.ascontiguousarray(series, dtype=np.int32)
    if xs.ndim == 0:
        raise ValueError("empty time series")
    elif xs.ndim == 1:
        n, m = 1, xs.size
    elif xs.ndim > 2:
        raise ValueError("dimension greater than 2")
    else:
        n, m = xs.shape

    if b == 0:
        b = max(2, np.amax(xs)+1)
    elif b < 1:
        raise ValueError("invalid base")

    tpm = np.zeros((b,b), dtype=np.float64)

    data = xs.ctypes.data_as(POINTER(c_int))
    tpm_data = tpm.ctypes.data_as(POINTER(c_double))

    e = ErrorCode(0)
    _inform_tpm(data, c_ulong(n), c_ulong(m), c_int(b), tpm_data, byref(e))
    error_guard(e)

    return tpm

_inform_tpm = _inform.inform_tpm
_inform_tpm.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_int, POINTER(c_double), POINTER(c_int)]
_inform_tpm.restype = POINTER(c_double)

