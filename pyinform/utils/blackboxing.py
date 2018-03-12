# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import byref, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def black_box(series, b=None, k=None, l=None, parts=None):
    series = np.ascontiguousarray(series, dtype=np.int32)

    if b is not None:
        b = np.ascontiguousarray(b, dtype=np.int64)
        if b.ndim != 1:
            b = b.flatten()

    if parts is None:
        return __black_box(series, b=b, k=k, l=l)
    else:
        if k is not None:
            raise ValueError('cannot provide both k and parts')
        if l is not None:
            raise ValueError('cannot provide both l and parts')
        return __black_box_parts(series, b=b, parts=parts)

def __black_box(series, b, k, l):
    if k is not None:
        k = np.ascontiguousarray(k, dtype=np.int64)
        if k.ndim != 1:
            k = k.flatten()
        if b is not None and b.shape != k.shape:
            raise ValueError('if provided, `b` and `k` must have the same shape')

    if l is not None:
        l = np.ascontiguousarray(l, dtype=np.int64)
        if l.ndim != 1:
            l = l.flatten()
        if b is not None and b.shape != l.shape:
            raise ValueError('if provided, `b` and `l` must have the same shape')
        if k is not None and k.shape != l.shape:
            raise ValueError('if provided, `k` and `l` must have the same shape')

    if series.ndim == 1:
        u, v, w = 1, 1, len(series)
        if b is None:
            b = np.ascontiguousarray([np.max(series) + 1], np.int32)
    elif series.ndim == 2 and b is not None:
        if len(b) == 1:
            u = 1
            v, w = series.shape
        else:
            v = 1
            u, w = series.shape
    elif series.ndim == 2 and k is not None:
        if len(k) == 1:
            u = 1
            v, w = series.shape
            b = np.ascontiguousarray([np.max(series) + 1], np.int32)
        else:
            v = 1
            u, w = series.shape
            b = np.ascontiguousarray(np.max(series, 1), dtype=np.int32) + 1
    elif series.ndim == 2 and l is not None:
        if len(l) == 1:
            u = 1
            v, w = series.shape
            b = np.ascontiguousarray([np.max(series) + 1], np.int32)
        else:
            v = 1
            u, w = series.shape
            b = np.ascontiguousarray(np.max(series, 1), dtype=np.int32) + 1
    elif series.ndim == 2:
        v = 1
        u, w = series.shape
        b = np.ascontiguousarray(np.max(series, 1), dtype=np.int32) + 1
    elif series.ndim == 3:
        u, v, w = series.shape
        if b is None:
            b = np.ascontiguousarray(list(map(np.max, series)), dtype=np.int32)
            b += 1
    else:
        raise ValueError('series dimension is too high')

    if len(b) != u:
        raise ValueError('shape mismatch: series and b are incompatible')
    if k is not None and len(k) != u:
        raise ValueError('shape mismatch: series and k are incompatible')
    if l is not None and len(l) != u:
        raise ValueError('shape mismatch: series and k are incompatible')

    if k is None:
        if l is None:
            out_len = w
        else:
            out_len = w - np.max(l)
    else:
        if l is None:
            out_len = w - np.max(k) + 1
        else:
            out_len = w - np.max(k) - np.max(l) + 1

    if v == 1:
        box = np.empty(out_len, dtype=np.int32)
    else:
        box = np.empty((v,out_len), dtype=np.int32)

    series_data = series.ctypes.data_as(POINTER(c_int))
    b_data = b.ctypes.data_as(POINTER(c_int))

    if k is None:
        k_data = None
    else:
        k_data = k.ctypes.data_as(POINTER(c_ulong))

    if l is None:
        l_data = None
    else:
        l_data = l.ctypes.data_as(POINTER(c_ulong))

    box_data = box.ctypes.data_as(POINTER(c_int))

    e = ErrorCode(0)

    _inform_black_box(series_data, c_ulong(u), c_ulong(v), c_ulong(w), b_data, k_data, l_data, box_data, byref(e))

    error_guard(e)

    return box

def __black_box_parts(series, b, parts):
    parts = np.ascontiguousarray(parts, dtype=np.int64)
    if parts.ndim != 1:
        parts = parts.flatten()
    if b is not None and b.shape != parts.shape:
        raise ValueError('if provided, `b` and `parts` must have the same shape')

    nparts = np.max(parts) + 1

    if series.ndim == 1:
        u, v, w = 1, 1, len(series)
        if b is None:
            b = np.ascontiguousarray([np.max(series) + 1], np.int32)
    elif series.ndim == 2 and b is not None:
        if len(b) == 1:
            u = 1
            v, w = series.shape
        else:
            v = 1
            u, w = series.shape
    elif series.ndim == 2:
        if len(parts) == 1:
            u = 1
            v, w = series.shape
            b = np.ascontiguousarray([np.max(series) + 1], np.int32)
        else:
            v = 1
            u, w = series.shape
            b = np.ascontiguousarray(np.max(series, 1), dtype=np.int32) + 1
    elif series.ndim == 3:
        u, v, w = series.shape
        if b is None:
            b = np.ascontiguousarray(list(map(np.max, series)), dtype=np.int32)
            b += 1
    else:
        raise ValueError('series dimension is too high')

    if len(b) != u:
        raise ValueError('shape mismatch: series and b are incompatible')
    if len(parts) != u:
        raise ValueError('shape mismatch: series and parts are incompatible')

    q = v * w 
    box = np.empty(nparts * (q + 1), dtype=np.int32)

    series_data = series.ctypes.data_as(POINTER(c_int))
    b_data = b.ctypes.data_as(POINTER(c_int))
    parts_data = parts.ctypes.data_as(POINTER(c_ulong))
    box_data = box.ctypes.data_as(POINTER(c_int))

    e = ErrorCode(0)

    _inform_black_box_parts(series_data, c_ulong(u), c_ulong(q), b_data, parts_data, c_ulong(nparts), box_data, byref(e))

    error_guard(e)

    if nparts == 1:
        bases = box[q:]
        if v == 1:
            box = box[:q]
        else:
            box = box[:q].reshape((v,w))
    else:
        bases = box[nparts * q:]
        if v == 1:
            box = box[:nparts * q].reshape((nparts,w))
        else:
            box = box[:nparts * q].reshape((nparts,v,w))

    return box, tuple(bases)

_inform_black_box = _inform.inform_black_box
_inform_black_box.argtypes = [POINTER(c_int), c_ulong, c_ulong, c_ulong, POINTER(c_int), POINTER(c_ulong), POINTER(c_ulong), POINTER(c_int), POINTER(c_int)]
_inform_black_box.restype = POINTER(c_int)

_inform_black_box_parts = _inform.inform_black_box_parts
_inform_black_box_parts.argtypes = [POINTER(c_int), c_ulong, c_ulong, POINTER(c_int), POINTER(c_ulong), c_ulong, POINTER(c_int), POINTER(c_int)]
_inform_black_box_parts.restype = POINTER(c_int)
