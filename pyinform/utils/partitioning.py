# Copyright 2016-2018 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""
.. testsetup::

    from pyinform.utils import black_box

Many analyses of complex systems consider partitioning of the system into
components or modules. One example of this is evidence of integration.

For generality, we represent a partitioning of :math:`N` items into :math:`1
\leq M \leq N` partitions as a sequence of integers :math:`p_1, \ldots, p_N`
where :math:`0 \leq p_i < M` is the partition to which the :math:`i`-th item
belongs.

As an example, suppose we partitioned :math:`\{X_1, X_2, X_3\}` as
:math:`\{\{X_1\}, \{X_2,X_3\}\}`. This would be represented as :math:`(0,1,1)`
because :math:`X_1` belongs to the zeroth partition and :math:`X_2,X_3` belong
to the first partition.

We procide the :py:func:`partitionings` generator to yield all unique
partitionings of a system of `N` elements.
"""
import numpy as np

from ctypes import c_int, c_ulong, POINTER
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard

def partitionings(n):
    """
    Yield the unique partitions of `n` things.

    **Example**: Say we wish to partition 3 items, :math:`\{X_1, X_2, X_3\}`.

    .. doctest::

        >>> list(partitionings(3))
        [array([0, 0, 0]), array([0, 0, 1]), array([0, 1, 0]), array([0, 1, 1]), array([0, 1, 2])]

    The result is the 5 unique partitioning schemes:

    .. math::

        \{\{X_1, X_2, X_3\}, \{\{X_1, X_2\}, \{X_3\}\}, \{\{X_1, X_3\}, \{X_2\}\},

    .. math::

        \{\{X_1\}, \{X_2, X_3\}\},\{\{X_1\}, \{X_2\}, \{X_3\}\}

    :param n: the number of items to partition
    :yield: a partitioning scheme
    """
    if n < 1:
        raise ValueError('argument must be positive, non-zero')

    parts = np.zeros(n, order='C', dtype=np.int64)
    yield parts.copy()

    parts_data = parts.ctypes.data_as(POINTER(c_ulong))

    nparts = _inform_next_partitioning(parts_data, c_ulong(n))
    while nparts != 0:
        yield parts.copy()
        nparts = _inform_next_partitioning(parts_data, c_ulong(n))

_inform_next_partitioning = _inform.inform_next_partitioning
_inform_next_partitioning.argtypes = [POINTER(c_ulong), c_ulong]
_inform_next_partitioning.restype = c_ulong
