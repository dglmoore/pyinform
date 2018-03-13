# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
.. testsetup::

    from pyinform.utils import black_box

It is often useful when analyzing complex systems to black-box components of
the system.  This process amounts to grouping components of the system and
treating them as a single entity, without regard for its small-scale structure.
As an example, consider you have two Boolean random variables :math:`X` and
:math:`Y`, and you observe time series of each simultaneously:

.. math::

    X:\ \{0,1,1,0,1,0,0,1\}\\\\
    Y:\ \{1,0,0,1,1,0,1,0\}

It may be worthwhile to consider these observations as observations of the
joint variable :math:`(X,Y)`:

.. math::

    (X,Y):\ \{(0,1),(1,0),(1,0),(0,1),(1,1),(0,0),(0,1),(1,0)\}.

The joint observations can then be naturally encoded, for example, as base-4
states

.. math::

    (X,Y):\ \{1,2,2,1,3,0,1,2\}.

We refer this process of mapping the observations of :math:`X` and :math:`Y` to
the encoded observations of stem:[(X,Y)] as black-boxing. In this case, the
black-boxing procedure is performed in "space" (you might think of :math:`X`
and :math:`Y` having locations in space).  We may also black-box in time. The
canonical example of this is considering the :math:`k`-history of a random
variable:

.. math ::

    X:\ \{0,1,1,0,1,0,0,1\}

.. math::

    X^{(2)}:\ \{((0,1),(1,1),(1,0),(0,1),(1,0),(0,0),(0,1)\},

and the observations of stem:[X^{(2)}] can be encoded as base-4 states:

.. math::

    X^{(2)}:\ \{(1,3,2,1,2,0,1\}.

We provide a black-boxing function that allows the user to black-box in both
space and into the future and past of a collection of frandom variables:
:py:func:`black_box`. The :py:func:`black_box` function also allows the user to
black-box based on a partitioning scheme (useful in the implementation of
integration measures such as evidence of integration).

Examples
--------

Structured Black Boxing
^^^^^^^^^^^^^^^^^^^^^^^

**Example 1**: Black-box two time series with no history or futures

.. doctest::

    >>> black_box([[0,1,1,0,1,0,0,1], [1,0,0,1,1,0,1,0]])
    array([1, 2, 2, 1, 3, 0, 1, 2], dtype=int32)

This is the first example described in :py:mod:`pyinform.utils.blackboxing`.

**Example 2**: Black-box a single time series in time with history length 2:

.. doctest::

    >>> black_box([0,1,1,0,1,0,0,1], k=2)
    array([1, 3, 2, 1, 2, 0, 1], dtype=int32)

This is the second example described in :py:mod:`pyinform.utils.blackboxing`.

*Example 3*: Black-box two time series with histories and futures:

In this example we consider two time series:

.. math::

    X:\ \{0,1,1,0,1,0,0,1\}\\\\
    Y:\ \{1,0,0,1,1,0,1,0\}

to produce observations of :math:`(X^{(2,0)},Y^{(1,1)})`

.. math::

    (X^{(2,0)},Y^{(1,1)}):\ \{(0,1,0,0),(1,1,0,1),(1,0,1,1),(0,1,1,0),(1,0,0,1),(0,0,1,0)\}

encoded as

.. math::

    (X^{(2,0)},Y^{(1,1)}):\ \{4,13,11,6,9,2\}.

.. doctest::

    >>> black_box([[0,1,1,0,1,0,0,1], [1,0,0,1,1,0,1,0]], k=(2,1), l=(0,1))
    array([ 4, 13, 11,  6,  9,  2], dtype=int32)

Partitionings
^^^^^^^^^^^^^

When the `parts` argument is provided to :py:func:`black_box`, a tuple is
returned. The first element is the black-boxed time series and the second is
a tuple of bases associated with the resulting time series.

**Example 1**: Black-box 4 time series into a single time series

.. doctest::

    >>> series = [[0,1,1,0,1,0,0,1,],[1,0,0,1,1,0,1,0],[0,0,0,1,1,1,0,0],[1,0,1,0,1,1,1,0]]
    >>> black_box(series, parts=(0,0,0,0))
    (array([ 5,  8,  9,  6, 15,  3,  5,  8], dtype=int32), (16,))

This could be done more simply as `black_box(series)`, but it is illustrative.

**Example 2**: Black-box 4 time series into two time series using the
partitioning scheme `(0,1,1,0)`. That is, combine the 0th and 4th, and 1st and
2nd.

.. doctest::

    >>> series = [[0,1,1,0,1,0,0,1,],[1,0,0,1,1,0,1,0],[0,0,0,1,1,1,0,0],[1,0,1,0,1,1,1,0]]
    >>> black_box(series, parts=(0,1,1,0))
    (array([[1, 2, 3, 0, 3, 1, 1, 2],
           [2, 0, 0, 3, 3, 1, 2, 0]], dtype=int32), (4, 4))

"""

import numpy as np

from ctypes import byref, c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def black_box(series, b=None, k=None, l=None, parts=None):
    """
    Black-box a collection of time series.

    History lengths for each time series may be provided through `r` and future
    lengths through `l`.

    Alternatively, the user may specify a `parts` argument which represents a
    partitioning scheme. In that case, neither `k` or `l` may be provided. See
    partitioning time series for more informatoin about partitioning schemes.

    Optionally, the user may provide a base for each of the provided time
    series; this is useful when the user knows the provided time series sample
    from a random variable with a given support, but not all states appear in
    the time series, e.g. if providing a time series `[0,1,0,1]` when the user
    knows that the base of the series should be `3` (`2` was never sampled).

    :param series: the time series
    :param b: the base of the time series (optional)
    :type b: int or a sequence
    :param k: the history lengths (optional)
    :type k: int or a sequence
    :param l: the future lengths (optional)
    :type l: int or a sequence
    :param parts: the partitioning scheme (optional)
    :type parts: a sequence
    :returns: the black-boxed time series (and the bases if `parts` is provided)
    """
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
