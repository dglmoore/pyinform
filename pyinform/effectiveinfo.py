# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_char_p, c_int, c_ulong, c_double, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def effective_info(tpm, inter=None):
    """
    Compute the effective information from a transition probability matrix and
    optional intervention distribution.

    If no intervention distribution is provided, a uniform intervention is used:

        >>> effective_info([[0.2, 0.8], [0.75, 0.25]])
        0.23159276936956244

    Of course, the user may choose to provide an intervention distribution:

        >>> effective_info([[0.2, 0.8], [0.75, 0.25]], inter=[0.25, 0.75])
        0.17422662802569278

    :param tpm: the transition probability matrix
    :param inter: the intervention distribution (optional)
    :return: the effective information (in bits)
    """
    tpm = np.ascontiguousarray(tpm, np.float64)

    if tpm.ndim != 2:
        raise ValueError('TPM is not a matrix')
    elif tpm.shape[0] != tpm.shape[1]:
        raise ValueError('TPM is not square')

    n = len(tpm)
    tpm_data = tpm.ctypes.data_as(POINTER(c_double))

    if inter is not None:
        inter = np.ascontiguousarray(inter, np.float64)
        if inter.ndim != 1:
            raise ValueError('intervention distribution is not a vector')
        elif inter.size != n:
            raise ValueError('intervention distribution and TPM are different sizes')
        inter_data = inter.ctypes.data_as(POINTER(c_double))
    else:
        inter_data = None

    e = ErrorCode(0)
    ei = _effective_info(tpm_data, inter_data, c_ulong(n), byref(e))
    error_guard(e)

    return ei

_effective_info = _inform.inform_effective_info
_effective_info.argtypes = [POINTER(c_double), POINTER(c_double), c_ulong, POINTER(c_int)]
_effective_info.restype = c_double
