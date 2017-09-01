.. _about:

========================================
PyGSP: Graph Signal Processing in Python
========================================

.. image:: https://readthedocs.org/projects/pygsp/badge/?version=latest
   :target: https://pygsp.readthedocs.io/en/latest/

.. image:: https://img.shields.io/travis/epfl-lts2/pygsp.svg
   :target: https://travis-ci.org/epfl-lts2/pygsp

.. image:: https://img.shields.io/coveralls/epfl-lts2/pygsp.svg
   :target: https://coveralls.io/github/epfl-lts2/pygsp

.. image:: https://img.shields.io/pypi/v/pygsp.svg
   :target: https://pypi.python.org/pypi/PyGSP

.. image:: https://img.shields.io/pypi/l/pygsp.svg
   :target: https://pypi.python.org/pypi/PyGSP

.. image:: https://img.shields.io/pypi/pyversions/pygsp.svg
   :target: https://pypi.python.org/pypi/PyGSP

.. image:: https://img.shields.io/github/stars/epfl-lts2/pygsp.svg?style=social
   :target: https://github.com/epfl-lts2/pygsp

The PyGSP is a Python package to ease `Signal Processing on Graphs
<https://arxiv.org/abs/1211.0053>`_
(a `Matlab counterpart <https://lts2.epfl.ch/gsp>`_
exists). It is a free software, distributed under the BSD license, and
available on `PyPI <https://pypi.python.org/pypi/PyGSP>`_. The
documentation is available on `Read the Docs
<https://pygsp.readthedocs.io>`_ and development takes place on `GitHub
<https://github.com/epfl-lts2/pygsp>`_.

This example demonstrates how to create a graph, a filter and analyse a signal on the graph.

>>> from pygsp import graphs, filters
>>> G = graphs.Logo()
>>> f = filters.Heat(G)
>>> Sl = f.filter(G.L.todense(), method='chebyshev')

Features
--------

This package facilitates graph constructions and give tools to perform signal processing on them.

A whole list of pre-constructed graphs can be used as well as core functions to create any other graph among which::

  - Neighest Neighbor Graphs
    - Bunny
    - Cube
    - Sphere
    - TwoMoons
    - ImgPatches
    - Grid2dImgPatches
  - Airfoil
  - BarabasiAlbert
  - Comet
  - Community
  - DavidSensorNet
  - ErdosRenyi
  - FullConnected
  - Grid2d
  - Logo GSP
  - LowStretchTree
  - Minnesota
  - Path
  - RandomRegular
  - RandomRing
  - Ring
  - Sensor
  - StochasticBlockModel
  - SwissRoll
  - Torus

On these graphs, filters can be applied to do signal processing. To this end, there is also a list of predefined filters on this toolbox::

  - Abspline
  - Expwin
  - Gabor
  - HalfCosine
  - Heat
  - Held
  - Itersine
  - MexicanHat
  - Meyer
  - Papadakis
  - Regular
  - Simoncelli
  - SimpleTf

Installation
------------

The PyGSP is available on PyPI::

    $ pip install pygsp

Note that you will need a recent version of ``pip``.
Please run ``pip install --upgrade pip`` if you get an installation error.

Contributing
------------

See the guidelines for contributing in ``CONTRIBUTING.rst``.

Acknowledgments
---------------

The PyGSP was started in 2014 as an academic open-source project for
research purpose at the `EPFL LTS2 laboratory <https://lts2.epfl.ch>`_.
This project has been partly funded by the Swiss National Science Foundation
under grant 200021_154350 "Towards Signal Processing on Graphs".
