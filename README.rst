=====
About
=====

PyGSP is a Graph Signal Processing Toolbox implemented in Python. It is a port of the Matlab GSP toolbox

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

System-wide installation::

    $ pip install pygsp

Installation in an isolated virtual environment::

    $ mkvirtualenv --system-site-packages pygsp
    $ pip install pygsp

You need virtualenvwrapper to run this command. The ``--system-site-packages``
option could be useful if you want to use a shared system installation of numpy, scipy and matplotlib. Their building and installation require quite some
dependencies.

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

License
-------
* Free software: GPL v3 license
