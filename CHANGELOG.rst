=========
Changelog
=========

All notable changes to this project will be documented in this file.
The format is based on `Keep a Changelog <https://keepachangelog.com>`_
and this project adheres to `Semantic Versioning <https://semver.org>`_.

Unreleased
----------

* ``print(graph)`` and ``print(filters)`` now show valuable information.
* Building a graph object is much faster.
* New rectangular filter (low-pass and band-pass).
* The exponential window has been updated from low-pass only to band-pass.
* Much better documentation for the coherence of the Fourier basis.
* Removed translate and modulate (they were not working and have no real use).
* Fixed and documented vertex-frequency transforms.
  They are now implemented as filter banks.
* Directed graphs are now completely supported.
* The differential operator (D, grad, div) is better tested and documented.
* ``G.dirichlet_energy(sig)`` computes the Dirichlet energy of a signal.
* Better documentation of the frame and its bounds.
* ``g.inverse()`` returns the pseudo-inverse of the filter bank.
* ``g.complement()`` returns the filter that makes the frame tight.
* Wave filter bank which application simulates the propagation of a wave.
* Continuous integration with Python 3.7, 3.8, 3.9. Dropped 2.7, 3.4, 3.5, 3.6.
* New implementation of the Sensor graph that is simpler and scales better.
* A new learning module with three functions to solve standard semi-supervised
  classification and regression problems.
* Import and export graphs and their signals to NetworkX and graph-tool.
* Save and load graphs and theirs signals to / from GraphML, GML, and GEXF.
* Documentation: path graph linked to DCT, ring graph linked to DFT.
* We now have a gallery of examples! That is convenient for users to get a
  taste of what the library can do, and to start working from a code snippet.
* Merged all the extra requirements in a single dev requirement.
* New star graph (implemented as a comet without a tail).

Experimental filter API (to be tested and validated):

* evaluate a filter bank with ``g(values)``
* filter with ``g(graph) @ signal``
* get an array representation (the frame) with ``g.toarray()``
* index the ``len(g)`` filters of a filter bank with ``g[idx]``
* concatenate filter banks with ``g + h``

Plotting:

The plotting interface was updated to be more user-friendly. First, the
documentation is now shown for ``filter.plot()``, ``graph.plot()``, and co.
Second, the API in the plotting library has been deprecated. That module is now
mostly for implementation only. Third, ``graph.plot()`` and
``graph.plot_signal()`` have been merged. As such, ``plot_signal()`` is
deprecated. Finally, the following parameter names were changed:

* ``plot_name`` => ``title``
* ``plot_eigenvalues`` => ``eigenvalues``
* ``show_sum`` => ``sum``
* ``show_edges`` => ``edges``
* ``vertex_size`` => ``size``
* ``npoints`` => ``n``
* ``save_as`` was removed

Other changes regarding plotting:

* Plotting functions return matplotlib figures and axes.
* Nodes, edges, and filters are plotted in transparency to avoid occlusion.
* The node index can be printed on top of nodes to identify them easily.
* Two vertex signals can now be plotted together as vertex color and size.
* Two edges signals can be plotted as edge color and width.
* Much faster (10 to 100 times faster) edge plotting with matplotlib.

There are many other small changes, look at the git history for the details.

0.5.1 (2017-12-15)
------------------

The focus of this release was to ease installation by not requiring
non-standard scientific Python packages to be installed.
It was mostly a maintenance release. A conda package is now available in
conda-forge. Moreover, the package can now be tried online thanks to binder.

The core functionality of this package only depends on numpy and scipy.
Dependencies which are only required for particular usages are included in the
alldeps extra dependency list. The alldeps list allows users to install
dependencies to enable all the features. Finally, those optional packages are
only loaded when needed, not when the PyGSP is imported. A nice side-effect is
that importing the PyGSP is now much faster!

The following packages were made optional dependencies:

* scikit-image, as it is only used to build patch graphs from images. The
  problem was that scikit-image does not provide a wheel for Windows and its
  build is painful and error-prone. Moreover, scikit-image has a lot of
  dependencies.
* pyqtgrpah, PyQt5 / PySide and PyOpenGl, as they are only used for interactive
  visualization, which not many users need. The problem was that pyqtgraph
  requires (via PyQt5, PySide, PyOpenGL) OpenGL (libGL.so) to be installed.
* matplotlib: while it is a standard package for any scientific or data science
  workflow, it's not necessary for users who only want to process data without
  plotting graphs, signals and filters.
* pyflann, as it is only used for approximate kNN. The problem was that the
  source distribution would not build for Windows. On conda-forge, (py)flann
  is not built for Windows either.

Moreover, matplotlib is now the default drawing backend. It's well integrated
with the Jupyter environment for scientific and data science workflows, and
most use cases do not require an interactive visualization. The pyqtgraph is
still available for interactivity.

0.5.0 (2017-10-06)
------------------

* Generalized the analysis and synthesis methods into the filter method.
* Signals are now rank-3 tensors of N_NODES x N_SIGNALS x N_FEATURES.
* Filter.evaluate returns a 2D array instead of a list of vectors.
* The differential operator was integrated in the Graph class, as the Fourier
  basis and the Laplacian were already.
* Removed the operators package. Transforms and differential operators went to
  the Graph class, the localization operator to the Filter class. These are now
  easier to use. Reduction became its own module.
* Graph object uses properties to delay the computation (lazy evaluation) of
  the Fourier basis (G.U, G.e, G.mu), the estimation of the largest eigenvalue
  (G.lmax) and the computation of the differential operator (G.D). A warning is
  issued if client code don't trigger computations with G.compute_*.
* Approximations for filtering have been moved in the filters package.
* PointCloud object removed. Functionality integrated in Graph object.
* data_handling module merged into utils.
* Fourier basis computed with eigh instead of svd (faster).
* estimate_lmax uses Lanczos instead of Arnoldi (symmetric sparse).
* Add a seed parameter to all non-deterministic graphs and filters.
* Filter.Nf indicates the number of filters in the filter bank.
* Don't check connectedness on graph creation (can take a lot of time).
* Erdos-Renyi now implemented as SBM with 1 block.
* Many bug fixes (e.g. Minnesota graph, Meyer filter bank, Heat filter, Mexican
  hat filter bank, Gabor filter bank).
* All GitHub issues fixed.

Plotting:

* Much better handling of plotting parameters.
* With matplotlib backend, plots are shown by default .
* Allows to set a default plotting backend as plotting.BACKEND = 'pyqtgraph'.
* qtg_default=False becomes backend='matplotlib'
* Added coordinates for path, ring, and randomring graphs.
* Set good default plotting parameters for most graphs.
* Allows to plot multiple filters in 1D with set_coordinates('line1D').
* Allows to pass existing matplotlib axes to the plotting functions.
* Show colorbar with matplotlib.
* Allows to set a 3D view point.
* Eigenvalues shown as vertical lines instead of crosses.
* Vertices can be highlighted, e.g. to show where filters where localized.

Documentation:

* More comprehensive documentation. Notably math definitions for operators.
* Most filters and graphs are plotted in the API documentation.
* List all methods and models at the top with autosummary.
* Useful package and module-level documentation.
* Doctests don't need to import numpy and the pygsp every time.
* Figures are automatically generated when building the documentation.
* Build on RTD with conda and matplotlib 2 (prettier plots).
* Intro and wavelets tutorials were updated.
* Reference guide is completely auto-generated from automodule.
* Added contribution guidelines.
* Documentation reorganization.
* Check that hyperlinks are valid.

Tests and infrastructure:

* Start test coverage analysis.
* Much more comprehensive tests. Coverage increased from 40% to 80%.
  Many bugs were uncovered.
* Always test with virtual X framebuffer to avoid the opening of lots of
  windows.
* Tested on Python 2.7, 3.4, 3.5, 3.6.
* Clean configuration files.
* Not using tox anymore (too painful to maintain multiple Pythons locally).
* Sort out installation of dependencies. Plotting should now work right away.
* Completely migrate development on GitHub.

0.4.2 (2017-04-27)
------------------

* Improve documentation.
* Various fixes.

0.4.1 (2016-09-06)
------------------

* Added routines to compute coordinates for the graphs.
* Added fast filtering of ideal band-pass.
* Implemented graph spectrograms.
* Added the Barabási-Albert model for graphs.
* Renamed PointClouds features.
* Various fixes.

0.4.0 (2016-06-17)
------------------

0.3.3 (2016-01-27)
------------------

* Refactoring graphs using object programming and fail safe checks.
* Refactoring filters to use only the Graph object used at the construction of the filter for all operations.
* Refactoring Graph pyramid to match MATLAB implementation.
* Removal of default coordinates (all vertices on the origin) for graphs that do not possess spatial meaning.
* Correction of minor issues on Python3+ imports.
* Various fixes.
* Finalizing demos for the documentation.

0.3.2 (2016-01-14)
------------------

0.3.1 (2016-01-12)
------------------

0.3.0 (2015-12-01)
------------------

0.2.1 (2015-10-20)
------------------

* Fix bug on pip installation.
* Update full documentation.

0.2.0 (2015-10-12)
------------------

* Adding functionalities to match the content of the Matlab GSP Box.
* First release of the PyGSP.

0.1.0 (2015-07-02)
------------------

* Main features of the box are present most of the graphs and filters can be used.
* The utils and operators modules also have most of their features implemented.
