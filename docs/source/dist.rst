.. _dist:

Empirical Distributions
=======================

The :py:class:`pyinform.dist.Dist` class provides an *empirical* distribution,
i.e. a histogram, representing the observed frequencies of some fixed-size set
of events. This class is the basis for all of the fundamental information
measures on discrete probability distributions.

Examples
--------

Example 1: Construction
^^^^^^^^^^^^^^^^^^^^^^^
You can construct a distribution with a specified number of unique observables.
This construction method results in an *invalid* distribution as no
observations have been made thus far.

.. doctest::

    >>> d = Dist(5)
    >>> d
    Dist.from_hist([0, 0, 0, 0, 0])
    >>> d.is_valid
    False
    >>> d.counts
    0
    >>> len(d)
    5

We also provide several static methods for constructing distributions in various
ways:

- From a list of observation counts (i.e. a histogram):

.. doctest::

    >>> d = Dist.from_hist([0,0,1,2,1,0,0])
    >>> d
    Dist.from_hist([0, 0, 1, 2, 1, 0, 0])
    >>> d.is_valid
    True
    >>> d.counts
    4
    >>> len(d)
    7

- Approximate from a list of probabilities:

.. doctest::

    >>> d = Dist.from_probs([0.25, 0.125, 0.625])
    >>> d
    Dist.from_hist([249999999, 124999999, 625000000])
    >>> d.is_valid
    True
    >>> d.counts
    999999998
    >>> len(d)
    3

- Estimate from a list of observed states:

.. doctest::

    >>> d = Dist.from_data([0,0,1,2,2,2,1,2])
    >>> d
    Dist.from_hist([2, 2, 4])
    >>> d.is_valid
    True
    >>> d.counts
    8
    >>> len(d)
    3

- A uniform distribution over a given support:

.. doctest::

    >>> d = Dist.uniform(4)
    >>> d
    Dist.from_hist([1, 1, 1, 1])
    >>> d.is_valid
    True
    >>> d.counts
    4
    >>> len(d)
    4

Example 2: Making Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a distribution has been constructed, we can begin making observations.
There are a few methods for doing so. The first uses the standard indexing
operations, treating the distribution similarly to a list:

.. doctest::

    >>> d = Dist(5)
    >>> for i in range(len(d)):
    ...     d[i] = i*i
    ... 
    >>> d
    Dist.from_hist([0, 1, 4, 9, 16])

The second method is to make *incremental* changes to the distribution. This
is useful when making observations of data as it is generated:

.. doctest::

    >>> obs = [1,0,1,2,2,3,2,3,2,2]
    >>> d = Dist(max(obs) + 1)
    >>> for x in obs:
    ...     d.tick(x)
    ... 
    1
    1
    2
    1
    2
    1
    3
    2
    4
    5
    >>> d
    Dist.from_hist([1, 2, 5, 2])

A third method essentially wrapps the above `for`-loop to provide a method of
recording data from a time time series:

.. doctest::

    >>> obs = [1,0,1,2,2,3,2,3,2,2]
    >>> d = Dist(max(obs) + 1)
    >>> d.accumulate(obs)
    10
    >>> d
    Dist.from_hist([1, 2, 5, 2])

Example 3: Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^

Once some observations have been made, we can start asking for probabilities.
As in the previous examples, there are multiple ways of doing this. The first
is to just ask for the probability of a given event.

.. doctest::

    >>> d = Dist.from_hist([3,0,1,2])
    >>> d.probability(0)
    0.5
    >>> d.probability(0)
    0.5
    >>> d.probability(1)
    0.0
    >>> d.probability(2)
    0.16666666666666666
    >>> d.probability(3)
    0.3333333333333333

Sometimes it is nice to just dump the probabilities out to an array:

.. doctest::

    >>> d = Dist.from_hist([3,0,1,2])
    >>> d.dump()
    array([0.5       , 0.        , 0.16666667, 0.33333333])

Example 4: Shannon Entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a distribution you can do lots of fun things with it. In
this example, we will compute the shannon entropy of a timeseries of
observed values.

.. doctest::

    >>> from math import log
    >>> d = Dist.from_data([1,0,1,2,2,1,2,3,2,2])
    >>> h = 0.0
    >>> for p in d.dump():
    ...     h -= p * log(p,2)
    ... 
    >>> h
    1.6854752972273344

Of course **PyInform** provides a function for this:
:py:func:`pyinform.shannon.entropy`.

.. doctest::

    >>> from pyinform.shannon import entropy
    >>> d = Dist.from_data([1,0,1,2,2,1,2,3,2,2])
    >>> entropy(d)
    1.6854752972273344

API Documentation
-----------------

.. automodule:: pyinform.dist

    .. autoclass:: pyinform.dist.Dist

        .. automethod::  pyinform.dist.Dist.__init__

        .. automethod::  pyinform.dist.Dist.from_hist

        .. automethod::  pyinform.dist.Dist.from_probs

        .. automethod::  pyinform.dist.Dist.from_data

        .. automethod::  pyinform.dist.Dist.uniform

        .. automethod::  pyinform.dist.Dist.__len__

        .. automethod::  pyinform.dist.Dist.__getitem__

        .. automethod::  pyinform.dist.Dist.__setitem__

        .. automethod::  pyinform.dist.Dist.resize

        .. automethod::  pyinform.dist.Dist.copy

        .. autoattribute::  pyinform.dist.Dist.counts

        .. autoattribute::  pyinform.dist.Dist.is_valid

        .. automethod::  pyinform.dist.Dist.tick

        .. automethod::  pyinform.dist.Dist.accumulate

        .. automethod::  pyinform.dist.Dist.probability

        .. automethod::  pyinform.dist.Dist.dump

