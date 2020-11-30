===================
Spherical samplings
===================

The `sphere`_ is ubiquitous: as a model of the Earth for weather and climate
modeling or geodesy (in geophysics), as the surface where measurements from an
observer of the universe are projected (in astrophysics), or as a model of the
brain's surface (in neuroscience).
Spherical data have been researched for decades, making the sphere the most
studied `manifold`_ after Euclidean spaces.

To represent and use spherical data (measured or simulated) on a computer, one
needs a finite representation of that data, hence a *sampling scheme* (also
called a *discretization* or *pixelization* of the sphere).
Our goal is to represent sampled spheres as graphs, and data as graph signals.
Analysis (and `learning`_) with spherical data then becomes a special case of
that with graph data; similarly to how Euclidean data is data on a grid graph
(e.g., :class:`~pygsp.graphs.Path`, :class:`~pygsp.graphs.Ring`,
:class:`~pygsp.graphs.Grid2d`, :class:`~pygsp.graphs.Torus`).
Unlike Euclidean space however, the spherical geometry is more difficult to
capture as it doesn't admit a uniform sampling.
The advantage of this approach---apart from its generality---is that it allows
a trade-off between the accurate exploitation of that geometry and
computational efficiency [DS2]_.

.. _sphere: https://en.wikipedia.org/wiki/Sphere
.. _manifold: https://en.wikipedia.org/wiki/Manifold
.. _learning: https://github.com/DeepSphere

Competing desiderata
--------------------

As there is no uniform sampling of the sphere, there are a number of competing
desiderata for a sampling scheme to fulfill.
Albeit for different reasons, the situation is similar to the problem of
`mapping the Earth`_ in cartography.

**Sampling theorem.**
An objective is to exactly represent a class of functions (often band-limited
functions) on the finite number of samples specified by the scheme.
This allows a `quadrature rule`_ to exactly integrate those functions and, more
specifically, allows for exact spherical harmonic transforms (SHTs) by
convolutions with `spherical harmonics`_.
A related goal is to represent this class of functions on the smallest number
of samples.
For example, the class of functions band-limited to degree :math:`L` has
:math:`L^2` degrees of freedom in harmonic space
(i.e., :math:`|\{ a_{ℓ,m} | -ℓ<m<ℓ, ℓ<L \}| = L^2`),
hence they should in principle be representable by :math:`L^2` samples.

**Equiarea.**
Another objective is for every sample to represent the same area of the sphere.
That is necessary for a sampling scheme to be asymptotically uniform, as some
areas would otherwise be over- or under-sampled.
It's also convenient as white noise from sensor acquisition remains white for
analysis.

**Hierarchical.**
The objective is for a high resolution discretization to build on the samples
of a lower resolution one.
That is useful to store, retrieve, analyze, or visualize maps at varying
resolutions.

**Fast SHT.**
A naive implementation of the SHT costs :math:`O(n^2)`, where :math:`n` is the
number of samples (and vertices of the graph).
The most common trick to reduce that cost to :math:`O(n^{3/2})` is to separate
the integral over latitude and longitude, and to leverage an `FFT`_ over
longitude.
This trick requires a scheme whose samples are arranged on rings of constant
latitudes.

.. _mapping the Earth: https://en.wikipedia.org/wiki/Map_projection
.. _quadrature rule: https://en.wikipedia.org/wiki/Numerical_integration
.. _spherical harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics
.. _FFT: https://en.wikipedia.org/wiki/Fast_Fourier_transform

Families of sampling schemes
----------------------------

Sampling schemes have been developed to make trade-offs on the desiderata that
are best aligned with their intended usage.
We can discern some families:

1. schemes designed around a sampling (quadrature) theorem,
2. schemes based on the subdivision of regular polyhedra,
3. schemes whose samples are positioned to minimize an energy function [BHS]_,
4. `space-filling curves`_.

We implemented four of the most common schemes used in spherical data analysis.
They belong to the first two families.

Another scheme, sampling uniformly at random, is implemented in
:class:`~pygsp.graphs.SphereRandom`, perhaps as a model of real measurements on
the Earth, where sensors (like `weather stations and boys`_) are not arranged
as a controlled sampling scheme.

.. _space-filling curves: https://en.wikipedia.org/wiki/Space-filling_curve
.. _weather stations and boys: https://en.wikipedia.org/wiki/Weather_station

Equiangular sampling (:class:`~pygsp.graphs.SphereEquiangular`)
---------------------------------------------------------------

The sphere is sampled at rings of constant latitudes with an equal angle
between them.
Each ring is made of an equal number of samples.
Samples are hence separated by constant latitudinal and longitudinal angles.

It is the most common sampling scheme of the sphere (used in geophysics,
astrophysics, as `UV sphere`_ in computer graphics) because (1) it is simple to
implement and represent as a 2D array and (2) it has a sampling theorem.
It is hierarchical but not equiarea (polar regions are heavily over-sampled).

Signals with bandwidth (band-limit) :math:`L` (polynomials of degree less than
:math:`L`) are exactly represented on that scheme with :math:`n=O(L^2)` samples
(vertices), while the exact number varies for each sampling theorem (with a
minimum of :math:`L^2`, the degrees of freedom).
Examples are:

* ``nlat=2*L, nlon=2*L, poles={0,1}`` (Driscoll--Healy) [DH1]_ [DH2]_,
* ``nlat=2*L-1, nlon=2*L-1, poles=0`` [Sk]_ ,
* ``nlat=2*L-1, nlon=2*L, poles=2`` (Clenshaw-Curtis quadrature) [KP]_,
* ``nlat=L, nlon=2*L-1, poles=1`` [MW]_.

Most induced SHTs cost :math:`O(L^3)=O(n^{3/2})`, while the Driscoll--Healy one
costs :math:`O(L^2 \log^2 L)`, albeit requiring pre-computations and storage.

.. _UV sphere: https://en.wikipedia.org/wiki/UV_mapping

Gauss--Legendre quadrature (:class:`~pygsp.graphs.SphereGaussLegendre`)
-----------------------------------------------------------------------

The sphere is sampled at rings of constant latitudes, where the latitudes
are given by the zeros of the Legendre polynomial.
The number of samples per ring is constant (``nlon``) or reduced towards the
poles (``nlon='reduction-scheme-name'``).

The scheme is used in geophysics and astrophysics for its sampling theorem and
:math:`O^{3/2}` SHT, while requiring less samples than the equiangular scheme.
An exact `Gauss--Legendre quadrature`_ for signals with bandwidth (band-limit)
:math:`L` (polynomials of degree less than :math:`L`) requires ``nlat=L`` and
``nlon=2*nlat=2*L`` [KP]_.
It is neither hierarchical nor equiarea (though reduced schemes help).

.. _Gauss--Legendre quadrature: https://en.wikipedia.org/wiki/Gaussian_quadrature

Subdivision of the icosahedron (:class:`~pygsp.graphs.SphereIcosahedral`)
-------------------------------------------------------------------------

The sampling is made of an `icosahedron`_ (a `regular and convex polyhedron`_
made of 12 vertices and 20 triangular faces) whose faces are subdivided into
:math:`m^2` triangles projected to the sphere.
The result is a :math:`\{3,5+\}_{m,0}` `geodesic polyhedron`_ (made of
:math:`20⋅m^2` triangles, :math:`n=10⋅m^2+2` vertices) or its dual, a
:math:`\{5+,3\}_{m,0}` `Goldberg polyhedron`_ (made of :math:`10⋅m^2-10`
hexagons, 12 pentagons, :math:`n=20⋅m^2` vertices).
The resulting `polyhedral graph`_ (i.e., a 3-vertex-connected planar graph) is
the 1-`skeleton`_ of the polyhedron.
All have `icosahedral symmetry`_.

The sampling is used in computer graphics (known as an icosphere) and
weather and climate modeling (known as a `geodesic grid`_) because edges and
faces are of approximately equal length and area.
It is hierarchical by definition but doesn't have a sampling theorem nor a fast
SHT.

.. _icosahedron: https://en.wikipedia.org/wiki/Icosahedron
.. _regular and convex polyhedron: https://en.wikipedia.org/wiki/Platonic_solid
.. _geodesic polyhedron: https://en.wikipedia.org/wiki/Geodesic_polyhedron
.. _Goldberg polyhedron: https://en.wikipedia.org/wiki/Goldberg_polyhedron
.. _polyhedral graph: https://en.wikipedia.org/wiki/Polyhedral_graph
.. _skeleton: https://en.wikipedia.org/wiki/N-skeleton
.. _geodesic grid: https://en.wikipedia.org/wiki/Geodesic_grid
.. _icosahedral symmetry: https://en.wikipedia.org/wiki/Icosahedral_symmetry

Subdivision of the cube (:class:`~pygsp.graphs.SphereCubed`)
------------------------------------------------------------

The sampling is made of a cube whose faces are subdivided into :math:`m^2`
finer quadrilaterals projected to the sphere.
The result is a convex polyhedron made of :math:`n=6⋅m^2` faces.
The graph vertices represent the quadrilateral faces.

The sampling is used in weather and climate modeling because edges and faces
are of approximately equal length and area (better when
``spacing='equiangular'``). It is hierarchical (for subdivisions arranged in
powers of 2) but doesn't have a sampling theorem nor a fast SHT.

Subdivision of the rhombic dodecahedron (:class:`~pygsp.graphs.SphereHealpix`)
------------------------------------------------------------------------------

The sampling is made of a `rhombic dodecahedron`_ (a convex polyhedron made of
12 rhombic faces) whose faces are subdivided into :math:`m^2` finer
quadrilaterals projected to the sphere.
The result is a convex polyhedron made of :math:`n=12⋅m^2` faces.
The graph vertices represent the quadrilateral faces.

The Hierarchical Equal Area isoLatitude Pixelisation (`HEALPix`_) [Go]_ was
developed for `cosmic microwave background (CMB)`_ maps but also used for other
astrophysical observations.
It is available as a `software package`_ with a `Python wrapper`_.
The scheme is hierarchical and equiarea, has a fast SHT, but doesn't have a
sampling theorem.

.. _rhombic dodecahedron: https://en.wikipedia.org/wiki/Rhombic_dodecahedron
.. _HEALPix: https://healpix.jpl.nasa.gov
.. _software package: https://healpix.sourceforge.io
.. _Python wrapper: https://healpy.readthedocs.io
.. _cosmic microwave background (CMB): https://en.wikipedia.org/wiki/Cosmic_microwave_background

Graphs from sampled spheres
---------------------------

A graph representing a sampled manifold must capture its topology (neighborhood
information) and geometry (metric/distance information).
The topology is captured by the (binary presence or absence of) edges.
The geometry is captured by vertex and edge weights---which can be interpreted
as `metric tensors`_ or quadrature weights.
Our goal is to construct a graph such that the operations, like Fourier
transforms and (learned) filtering, on the discrete structure correspond to
their equivalent on the continuous sphere [DS1]_.
Equivalently, the symmetries of the sphere shall correspond to the
automorphisms of the graph.

The first choice to be made is the number of edges to include.
There are :math:`O(n^2)` relations (between every pair of vertices) to capture.
While a `complete graph`_ will capture that (the graph Laplacian is known to
asymptotically converge to the Laplace--Beltrami operator in that case), it is
computationally costly: The :math:`O(n^2)` edges incur an :math:`O(n^2)` cost
to filtering.
We hence desire sparse graphs to strike a balance between efficiency and the
accurate representation of the sphere's geometry [DS2]_.
Accuracy is reduced when sparsifying because degrees of freedom are lost when
edges are removed.
Consider how the relation between two vertices is controlled: If they are
connected by an edge, that edge's weight directly control their relation; but
if they are not, indirect connections through other vertices determine their
relation.
There are diminishing returns in adding edges.

Once the sparsity is chosen (via :class:`~pygsp.graphs.NNGraph` parameters),
the difficulty is in setting the weights.
Optimal edge weights have been sought
in [DS2]_ for :class:`~pygsp.graphs.SphereHealpix` and
in [KF]_ for :class:`~pygsp.graphs.SphereEquiangular`
(albeit for latitudinal and longitudinal rotations only).
A way to set the weights for any sampling (and edge sparsity) of the sphere has
yet to be found.

.. _metric tensors: https://en.wikipedia.org/wiki/Metric_tensor
.. _complete graph: https://en.wikipedia.org/wiki/Complete_graph

References
----------

.. [BHS] S. V. Borodachov, D. P. Hardin and E. B. Saff, Discrete energy on
   rectifiable sets, 2019.

.. [DH1] J. R. Driscoll et D. M. Healy, Computing Fourier Transforms and
   Convolutions on the 2-Sphere, 1994.
.. [DH2] D. M. Healy et al., FFTs for the 2-Sphere-Improvements and
   Variations, 2003.
.. [Sk] W. Skukowsky, A quadrature formula over the sphere with application
   to high resolution spherical harmonic analysis, 1986.
.. [KP] J. Keiner and D. Potts, Fast evaluation of quadrature formulae on
   the sphere, 2008.
.. [MW] J. D. McEwen and Y. Wiaux, A novel sampling theorem on the sphere,
   2011.

.. [Go] K. M. Gorski et al., HEALPix: a Framework for High Resolution
   Discretization and Fast Analysis of Data Distributed on the Sphere,
   2005.

.. [KF] R. Khasanova and P. Frossard, Graph-Based Classification of
   Omnidirectional Images, 2017.
.. [DS1] M. Defferrard et al., DeepSphere: towards an equivariant graph-based
   spherical CNN, 2019.
.. [DS2] M. Defferrard et al., DeepSphere: a graph-based spherical CNN, 2020.
