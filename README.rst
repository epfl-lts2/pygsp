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

* Documentation: https://pygsp.readthedocs.io
* Development: https://github.com/epfl-lts2/pygsp
* Matlab counterpart: https://github.com/epfl-lts2/gspbox

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

Contributing
------------

See :ref:`contributing`.

Authors
-------

* Basile Châtillon <basile.chatillon@epfl.ch>,
* Alexandre Lafaye <alexandre.lafaye@epfl.ch>,
* Lionel Martin <lionel.martin@epfl.ch>,
* Nicolas Rod <nicolas.rod@epfl.ch>,
* Rodrigo Pena <rodrigo.pena@epfl.ch>
* Michaël Defferrard <michael.defferrard@epfl.ch>

Acknowledgment
--------------

This project has been partly funded by the Swiss National Science Foundation under grant 200021_154350 "Towards Signal Processing on Graphs".
