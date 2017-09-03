=======
History
=======

x.x.x (xxxx-xx-xx)
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
* Add a seed parameter to various graphs and filters.
* Filter.Nf indicates the number of filters in the filter bank.
* Don't check connectedness on graph creation (can take a lot of time).
* Show plots by default with matplotlib backend.
* Many bug fixes (e.g. Minnesota graph, Meyer filter bank, Heat filter, Mexican
  hat filter bank, Gabor filter bank).
* All GitHub issues fixed.

Plotting:

* Much better handling of plotting parameters.
* Allows to set a default plotting backend as plotting.BACKEND = 'pyqtgraph'.
* qtg_default=False becomes backend='matplotlib'
* Added coordinates for path, ring, and randomring graphs.
* Set good default plotting parameters for most graphs.
* Allows to plot multiple filters in 1D with set_coordinates('line1D').
* Allows to pass existing matplotlib axes to the plotting functions.
* Show colorbar with matplotlib.
* Allows to set a 3D view point.

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
