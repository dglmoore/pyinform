# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

from ctypes import c_bool, c_double, c_uint, c_ulong, c_void_p, POINTER
from pyinform import _inform

class Dist:
    """
    Dist is class designed to represent empirical probability distributions,
    i.e. histograms, for cleanly logging observations of time series data.

    The premise behind this class is that it allows **PyInform** to define
    the standard entropy measures on distributions. This reduces functions
    such as :py:func:`pyinform.activeinfo.active_info` to building
    distributions and then applying standard entropy measures.
    """
    def __init__(self, n=None, pointer=None):
        """
        Construct a distribution

        .. rubric:: Examples:

        ::

            >>> Dist(5)
            <pyinform.dist.Dist instance at 0x7f0c1f6a77a0>

        :param int n: the size of the distributions's support
        :raises ValueError: ``n <= 0``
        :return: The distribution
        """

        if n is None and pointer is None:
            raise ValueError("must provide a support")
        elif n is not None and pointer is not None:
            raise ValueError("")
        elif pointer is not None:
            self._dist = pointer
        elif n == 0:
            raise ValueError("support is zero")
        elif n < 0:
            raise ValueError("support is negative")
        else:
            self._dist = _dist_alloc(c_ulong(n))
            if not self._dist:
                raise MemoryError()

    @classmethod
    def from_hist(cls, hist):
        """
        Create a distribution from a histogram

        .. rubric:: Examples:

        ::

            >>> d = Dist.from_hist([0,1,2,1])
            >>> d[:]
            [0, 1, 2, 1]

        :param hist: the counts for each bin
        :return: the distribution
        """
        hist = np.ascontiguousarray(hist, dtype=np.uint32)
        if hist.ndim == 0:
            raise ValueError("histogram is zero-dimensional")
        elif hist.ndim > 1:
            raise ValueError("histogram is multi-dimensional")
        elif hist.size == 0:
            raise ValueError("support is empty")
        data = hist.ctypes.data_as(POINTER(c_uint))
        dist = _dist_create(data, hist.size)
        return Dist(pointer=dist)

    @classmethod
    def from_probs(cls, probs, tol=1e-9):
        """
        Create a distribution from probabilities

        .. rubric:: Examples:

        ::

            >>> d = Dist.from_probs([0.2, 0.4, 0.4])
            >>> d[:]
            [1, 2, 2]
            >>> d = Dist.from_probs([0.25, 0.125, 0.625])
            >>> d[:]
            [249999999, 124999999, 625000000]
            >>> d = Dist.from_probs([0.625, 0.375])
            >>> d[:]
            [5, 3]

        :param probs: the probability of each event
        :param tol:   the acceptable tolerance
        :return: the distribution
        """
        probs = np.ascontiguousarray(probs, dtype=np.float64)
        if probs.size == 0:
            raise ValueError("no probabilities provided")
        elif np.any(probs < 0.0):
            raise ValueError("negative probability provided")
        elif np.abs(np.sum(probs) - 1.0) > tol:
            raise ValueError("probabilities must sum to 1.0")
        data = probs.ctypes.data_as(POINTER(c_double))
        dist = _dist_approx(data, probs.size, tol)
        return Dist(pointer=dist)

    @classmethod
    def from_data(cls, seq, n=None):
        """
        Create a distribution from a sequence of observations

        .. rubric:: Examples:

        ::

            >>> d = Dist.from_data([0,1,1,1,0,1,1])
            >>> d[:]
            [2, 5]
            >>> d = Dist.from_data([0,1,1,1,0,1,1], n=4)
            >>> d[:]
            [2, 5, 0, 0]

        :param seq: a sequence of observed events
        :param n:   the minimum size of the support
        :return: the inferred distribution
        """
        seq = np.ascontiguousarray(seq, dtype=np.uint32)
        if seq.size == 0:
            raise ValueError("no data in sequence")
        data = seq.ctypes.data_as(POINTER(c_uint))
        dist = Dist(pointer=_dist_infer(data, seq.size))
        if n is not None and len(dist) < n:
            dist.resize(n)
        return dist

    @classmethod
    def uniform(cls, n):
        """
        Create a uniform distribution of a given size

        .. rubric:: Examples:

        ::

            >>> d = Dist.uniform(2)
            >>> d[:]
            [1, 1]
            >>> d = Dist.uniform(5)
            >>> d[:]
            [1, 1, 1, 1, 1]

        :param n: the size of the support
        :return: the uniform distribution
        """
        if n == 0:
            raise ValueError("support is zero")
        elif n < 0:
            raise ValueError("support is negative")
        return Dist(pointer=_dist_uniform(n))

    def __dealloc__(self):
        """
        Deallocate the memory underlying the distribution.
        """
        if self._dist:
            _dist_free(self._dist)

    def __len__(self):
        """
        Determine the size of the support of the distribution.

        .. rubric:: Examples:

        ::

            >>> len(Dist(5))
            5
            >>> len(Dist.from_hist([0,1,5]))
            3

        See also :py:attr:`.counts`.

        :return: the size of the support
        :rtype: int
        """
        return int(_dist_size(self._dist))

    def resize(self, n):
        """
        Resize the support of the distribution in place.

        If the distribution...

        - **shrinks** - the last ``len(self) - n`` elements are lost, the rest are preserved
        - **grows** - the last ``n - len(self)`` elements are zeroed
        - **is unchanged** - well, that sorta says it all, doesn't it?

        .. rubric:: Examples:

        ::

            >>> d = Dist(5)
            >>> d.resize(3)
            >>> len(d)
            3
            >>> d.resize(8)
            >>> len(d)
            8

        ::

            >>> d = Dist.from_hist([1,2,3,4])
            >>> d.resize(2)
            >>> list(d)
            [1, 2]
            >>> d.resize(4)
            >>> list(d)
            [1, 2, 0, 0]

        :param int n: the desired size of the support
        :raises ValueError: if the requested size is zero
        :raises MemoryError: if memory allocation fails in the C call
        """
        if n <= 0:
            raise ValueError("support is zero")
        self._dist = _dist_realloc(self._dist, c_ulong(n))
        if not self._dist:
            raise MemoryError()

    def copy(self):
        """
        Perform a deep copy of the distribution.

        .. rubric:: Examples:

        ::

            >>> d = Dist.from_hist([1,2,3])
            >>> e = d
            >>> e[0] = 3
            >>> list(e)
            [3, 2, 3]
            >>> list(d)
            [3, 2, 3]

        ::

            >>> f = d.copy()
            >>> f[0] = 1
            >>> list(f)
            [1, 2, 3]
            >>> list(d)
            [3, 2, 3]

        :returns: the copied distribution
        :rtype: :py:class:`pyinform.dist.Dist`
        """
        d = Dist(len(self))
        _dist_copy(self._dist, d._dist)
        return d

    @property
    def counts(self):
        """
        Return the number of observations made thus far.

        .. rubric:: Examples:

        ::

            >>> d = Dist(5)
            >>> d.counts
            0

        ::

            >>> d = Dist.from_hist([1,0,3,2])
            >>> d.counts
            6

        See also :py:meth:`.__len__`.

        :return: the number of observations
        :rtype: int
        """
        return _dist_counts(self._dist)

    @property
    def is_valid(self):
        """
        Determine if the distribution is a valid probability distribution, i.e.
        if the support is not empty and at least one observation has been made.

        .. rubric:: Examples:

        ::

            >>> d = Dist(5)
            >>> d.is_valid
            False

        ::

            >>> d = Dist.from_hist([0,0,0,1])
            >>> d.is_valid
            True

        See also :py:meth:`.__len__` and :py:attr:`.counts`.

        :return: a boolean signifying that the distribution is valid
        :rtype: bool
        """
        return _dist_is_valid(self._dist)

    def __getitem__(self, event):
        """
        Get the number of observations made of *event*.

        .. rubric:: Examples:

        ::

            >>> d = Dist(2)
            >>> (d[0], d[1])
            (0, 0)

        ::

            >>> d = Dist.from_hist([0,1])
            >>> (d[0], d[1])
            (0, 1)

        See also :py:meth:`.__setitem__`, :py:meth:`.tick` and
        :py:meth:`.probability`.

        :param int event: the observed event
        :return: the number of observations of *event*
        :rtype: int
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if isinstance(event, slice):
           return [ self[i] for i in range(*event.indices(len(self))) ]
        elif event < 0 or event >= len(self):
            raise IndexError()
        return int(_dist_get(self._dist, c_ulong(event)))

    def __setitem__(self, event, value):
        """
        Set the number of observations of *event* to *value*.

        If *value* is negative, then the observation count is set to zero.

        .. rubric:: Examples:

        ::

            >>> d = Dist(2)
            >>> for i, _ in enumerate(d):
            ...     d[i] = i*i
            ...
            >>> list(d)
            [0, 1]

        ::

            >>> d = Dist.from_hist([0,1,2,3])
            >>> for i, n in enumerate(d):
            ...     d[i] = 2 * n
            ...
            >>> list(d)
            [0, 2, 4, 6]


        See also :py:meth:`.__getitem__` and :py:meth:`.tick`.

        :param int event: the observed event
        :param int value: the number of observations
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        value = max(0, value)
        return _dist_set(self._dist, c_ulong(event), c_uint(value))

    def tick(self, event):
        """
        Make a single observation of *event*, and return the total number
        of observations of said *event*.

        .. rubric:: Examples:

        ::

            >>> d = Dist(5)
            >>> for i, _ in enumerate(d):
            ...     assert(d.tick(i) == 1)
            ...
            >>> list(d)
            [1, 1, 1, 1, 1]

        ::

            >>> d = Dist.from_hist([0,1,2,3])
            >>> for i, _ in enumerate(d):
            ...     assert(d.tick(i) == i + 1)
            ...
            >>> list(d)
            [1, 2, 3, 4]

        See also :py:meth:`.__getitem__` and :py:meth:`.__setitem__`.

        :param int event: the observed event
        :return: the total number of observations of *event*
        :rtype: int
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if event < 0 or event >= len(self):
            raise IndexError()
        return _dist_tick(self._dist, c_ulong(event))

    def accumulate(self, events):
        """
        Accumulate events into a distribution

        .. rubric:: Examples:

        ::

            >>> d = Dist(3)
            >>> d[:]
            [0, 0, 0]
            >>> d.accumulate([0,1,1,0,2,1,2,2,1,2])
            10
            >>> d[:]
            [2, 4, 4]

        :param events: a sequence of observed events
        :return: the number of events recorded
        """
        events = np.ascontiguousarray(events, dtype=np.uint32)
        data = events.ctypes.data_as(POINTER(c_uint))
        n = _dist_accumulate(self._dist, data, len(events))
        if n != len(events):
            raise IndexError()
        return int(n)

    def probability(self, event):
        """
        Compute the empiricial probability of an *event*.

        .. rubric:: Examples:

        ::

            >>> d = Dist.uniform(4)
            >>> for i, _ in enumerate(d):
            ...     assert(d.probability(i) == 1./4)
            ...

        See also :py:meth:`.__getitem__` and :py:meth:`.dump`.

        :param int event: the observed event
        :return: the empirical probability *event*
        :rtype: float
        :raises ValueError: if ``not self.is_valid``
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if not self.is_valid:
            raise ValueError("invalid distribution")
        elif event < 0 or event >= len(self):
            raise IndexError()
        return _dist_prob(self._dist, c_ulong(event))

    def dump(self):
        """
        Compute the empirical probability of each observable event and return
        the result as an array.

        .. rubric:: Examples:

        ::

            >>> d = Dist.from_hist([1,2,2,1])
            >>> d.dump()
            array([ 0.16666667,  0.33333333,  0.33333333,  0.16666667])

        See also :py:meth:`.probability`.

        :return: the empirical probabilities of all o
        :rtype: ``numpy.ndarray``
        :raises ValueError: if ``not self.is_valid``
        :raises RuntimeError: if the dump fails in the C call
        :raises IndexError: if ``event < 0 or len(self) <= event``
        """
        if not self.is_valid:
            raise ValueError("invalid distribution")
        n = len(self)
        probs = np.empty(n, dtype=np.float64)
        data = probs.ctypes.data_as(POINTER(c_double))
        m = _dist_dump(self._dist, data, c_ulong(n))
        if m != n:
            raise RuntimeError("cannot dump the distribution")
        return probs

    def __repr__(self):
        """
        Return a `eval`-able string representation of the distribution.

        .. rubric:: Examples:

        ::

            >>> dist = Dist.from_hist([0,1,1,0,1])
            >>> dist
            Dist.from_hist([0, 1, 1, 0, 1])
            >>> dist = Dist.from_data([1,2,4])
            >>> dist
            Dist.from_hist([0, 1, 1, 0, 1])

        :return: an `eval`-able string
        """
        return 'Dist.from_hist({})'.format(self[:])

_dist_alloc = _inform.inform_dist_alloc
_dist_alloc.argtypes = [c_ulong]
_dist_alloc.restype = c_void_p

_dist_realloc = _inform.inform_dist_realloc
_dist_realloc.argtypes = [c_void_p, c_ulong]
_dist_realloc.restype = c_void_p

_dist_copy = _inform.inform_dist_copy
_dist_copy.argtypes = [c_void_p, c_void_p]
_dist_copy.restype = c_void_p

_dist_create = _inform.inform_dist_create
_dist_create.argtypes = [POINTER(c_uint), c_ulong]
_dist_create.restype = c_void_p

_dist_infer = _inform.inform_dist_infer
_dist_infer.argtypes = [POINTER(c_uint), c_ulong]
_dist_infer.restype = c_void_p

_dist_approx = _inform.inform_dist_approximate
_dist_approx.argtypes = [POINTER(c_double), c_ulong, c_double]
_dist_approx.restype = c_void_p

_dist_uniform = _inform.inform_dist_uniform
_dist_uniform.argtypes = [c_ulong]
_dist_uniform.restype = c_void_p

_dist_free = _inform.inform_dist_free
_dist_free.argtypes = [c_void_p]
_dist_free.restype = None

_dist_size = _inform.inform_dist_size
_dist_size.argtypes = [c_void_p]
_dist_size.restype = c_ulong

_dist_counts = _inform.inform_dist_counts
_dist_counts.argtypes = [c_void_p]
_dist_counts.restype = c_uint

_dist_is_valid = _inform.inform_dist_is_valid
_dist_is_valid.argtypes = [c_void_p]
_dist_is_valid.restype = c_bool

_dist_get = _inform.inform_dist_get
_dist_get.argtypes = [c_void_p, c_ulong]
_dist_get.restype = c_uint

_dist_set = _inform.inform_dist_set
_dist_set.argtypes = [c_void_p, c_ulong, c_uint]
_dist_set.restype = c_uint

_dist_tick = _inform.inform_dist_tick
_dist_tick.argtypes = [c_void_p, c_ulong]
_dist_tick.restype = c_uint

_dist_prob = _inform.inform_dist_prob
_dist_prob.argtypes = [c_void_p, c_ulong]
_dist_prob.restype = c_double

_dist_dump = _inform.inform_dist_dump
_dist_dump.argtypes = [c_void_p, POINTER(c_double), c_ulong]
_dist_dump.restype = c_ulong

_dist_accumulate = _inform.inform_dist_accumulate
_dist_accumulate.argtypes = [c_void_p, POINTER(c_uint), c_ulong]
_dist_accumulate.restype = c_ulong
