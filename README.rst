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

* Graphs

  - Basic Graph structure
  - NNGraph
  - Bunny
  - Cube
  - Sphere
  - TwoMoons
  - Grid2d
  - Torus
  - Comet
  - LowStretchTree
  - RandomRegular
  - Ring
  - Community
  - Minnesota
  - Sensor
  - Airfoil
  - DavidSensorNet
  - FullConnected
  - Logo
  - Path
  - RandomRing

* Filters

  - Basic Filter structure
  - Abspline
  - Expwin
  - HalfCosine
  - Itersine
  - MexicanHat
  - Meyer
  - SimpleTf
  - Papadakis
  - Regular
  - Simoncelli
  - Held
  - Heat

Installation
------------

Ubuntu
^^^^^^
The PyGSP module is available on PyPI, the Python Package Index.
If you don't have pip, install it.::

    $ sudo apt-get install python-pip

Ideally, you should be able to install the PyGSP on your computer by simply entering the following command::

    $ sudo pip install pygsp

Unfortunately, this command will most likely fail. For some reason, the install will generally fail when it'll try to install requirements such as numpy and scipy. You'll need to install them manually by doing so::

    $ sudo pip install numpy
    $ sudo pip install scipy

For a classic UNIX system, you will need python-dev(el) (or equivalent) installed as a system package as well as the fortran extension for your favorite compiler (gfortran for gcc). You will also need the blas/lapack implementation for your system. If you can't install numpy or scipy, try installing the following and then install numpy and scipy::

    $ sudo apt-get install python-dev liblapack-dev libatlas-dev gcc gfortran

If you want to use the plotting functionalities of the PyGSP, you have to install matplotlib or pygtgraph. For matplotlib, just do::

    $ sudo apt-get python-matplotlib


Another way is to manually download from PyPI, unpack the package and install
with::

    $ python setup.py install

Execute the project test suite once to make sure you have a working install::

    $ python setup.py test

Authors
-------

* Basile Châtillon <basile.chatillon@epfl.ch>,
* Alexandre Lafaye <alexandre.lafaye@epfl.ch>,
* Nicolas Rod <nicolas.rod@epfl.ch>
