.. _about:

=====
About
=====

PyGSP is a Graph Signal Processing Toolbox implemented in Python. It is a port of the Matlab GSP toolbox.

.. image:: https://img.shields.io/travis/epfl-lts2/pygsp.svg
   :target: https://travis-ci.org/epfl-lts2/pygsp

* Development : https://github.com/epfl-lts2/pygsp
* GSP matlab toolbox : https://github.com/epfl-lts2/gspbox

Features
--------
This toolbox facilitate graph constructions and give tools to perform signal processing on them.

A whole list of preconstructed graphs can be used as well as core functions to create any other graph among which::

  - Neighest Neighbor Graphs
    - Bunny
    - Cube
    - Sphere
    - TwoMoons
  - Airfoil
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
  - Swiss roll
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

Ubuntu
^^^^^^
The PyGSP module is available on PyPI, the Python Package Index.
If you don't have pip, install it.::

    $ sudo apt-get install python-pip

Ideally, you should be able to install the PyGSP on your computer by simply entering the following command::

    $ pip install pygsp

This installation requires numpy and scipy. If you don't have them installed already, pip installing pygsp will try to install them for you. Note that these two mathematical libraries requires additional system packages.

For a classic UNIX system, you will need python-dev(el) (or equivalent) installed as a system package as well as the fortran extension for your favorite compiler (gfortran for gcc). You will also need the blas/lapack implementation for your system. If you can't install numpy or scipy, try installing the following and then install numpy and scipy::

    $ sudo apt-get install python-dev liblapack-dev libatlas-dev gcc gfortran

Then, try again to install the pygsp::
    
    $ pip install pygsp

Plotting
^^^^^^^^
If you want to use the plotting functionalities of the PyGSP, you have to install matplotlib or pygtgraph. For matplotlib, just do::

    $ sudo apt-get python-matplotlib


Another way is to manually download from PyPI, unpack the package and install with::

    $ python setup.py install

Instructions and requirements to install pyqtgraph can be found at http://www.pyqtgraph.org/.

Testing
^^^^^^^
Execute the project test suite once to make sure you have a working install::

    $ python setup.py test

Authors
-------

* Basile Châtillon <basile.chatillon@epfl.ch>,
* Alexandre Lafaye <alexandre.lafaye@epfl.ch>,
* Lionel Martin <lionel.martin@epfl.ch>,
* Nicolas Rod <nicolas.rod@epfl.ch>

Acknowledgment
--------------

This project has been partly funded by the Swiss National Science Foundation under grant 200021_154350 "Towards Signal Processing on Graphs".
