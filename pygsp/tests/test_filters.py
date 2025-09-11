"""
Test suite for the filters module of the pygsp package.

"""

import numpy as np
import pytest

from pygsp import filters, graphs


@pytest.fixture(scope="module")
def test_graph():
    """Graph for filter testing."""
    G = graphs.Sensor(123, seed=42)
    G.compute_fourier_basis()
    return G


@pytest.fixture(scope="module")
def rng():
    """Random number generator."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def test_signal(test_graph, rng):
    """Test signal for filtering."""
    return rng.uniform(size=test_graph.N)


def generate_coefficients(test_graph, N, Nf, vertex_delta=83):
    """Generate test coefficients."""
    S = np.zeros((N * Nf, Nf))
    S[vertex_delta] = 1
    for i in range(Nf):
        S[vertex_delta + i * test_graph.N, i] = 1
    return S


def _test_filter_methods(test_graph, test_signal, f, tight, check=True):
    """Test various filter methods."""
    assert f.G is test_graph

    f.evaluate(test_graph.e)
    f.evaluate(np.random.default_rng().uniform(0, 1, size=(4, 6, 3)))

    A, B = f.estimate_frame_bounds(test_graph.e)
    if tight:
        np.testing.assert_allclose(A, B)
    else:
        assert B - A > 0.01

    # Analysis.
    s2 = f.filter(test_signal, method="exact")
    s3 = f.filter(test_signal, method="chebyshev", order=100)

    # Synthesis.
    s4 = f.filter(s2, method="exact")
    s5 = f.filter(s3, method="chebyshev", order=100)

    if check:
        # Chebyshev should be close to exact.
        # Does not pass for Gabor, Modulation and Rectangular (not smooth).
        np.testing.assert_allclose(s2, s3, rtol=0.1, atol=0.01)
        np.testing.assert_allclose(s4, s5, rtol=0.1, atol=0.01)

    if tight:
        # Tight frames should not loose information.
        np.testing.assert_allclose(s4, A * test_signal)
        assert np.linalg.norm(s5 - A * test_signal) < 0.1

    # Computing the frame is an alternative way to filter.
    if f.n_filters != test_graph.n_vertices:
        # It doesn't work (yet) if the number of filters is the same as the
        # number of nodes as f.filter() infers that we want synthesis where
        # we actually want analysis.
        F = f.compute_frame(method="exact")
        s = F.dot(test_signal).reshape(-1, test_graph.N).T.squeeze()
        np.testing.assert_allclose(s, s2)

        F = f.compute_frame(method="chebyshev", order=100)
        s = F.dot(test_signal).reshape(-1, test_graph.N).T.squeeze()
        np.testing.assert_allclose(s, s3)


def test_filter(test_graph, rng, n_filters=5):
    """Test filter functionality."""
    g = filters.MexicanHat(test_graph, n_filters)

    s1 = rng.uniform(size=test_graph.N)
    s2 = s1.reshape((test_graph.N, 1))
    s3 = g.filter(s1)
    s4 = g.filter(s2)
    s5 = g.analyze(s1)
    assert s3.shape == (test_graph.N, n_filters)
    np.testing.assert_allclose(s3, s4)
    np.testing.assert_allclose(s3, s5)

    s1 = rng.uniform(size=(test_graph.N, 4))
    s2 = s1.reshape((test_graph.N, 4, 1))
    s3 = g.filter(s1)
    s4 = g.filter(s2)
    s5 = g.analyze(s1)
    assert s3.shape == (test_graph.N, 4, n_filters)
    np.testing.assert_allclose(s3, s4)
    np.testing.assert_allclose(s3, s5)

    s1 = rng.uniform(size=(test_graph.N, n_filters))
    s2 = s1.reshape((test_graph.N, 1, n_filters))
    s3 = g.filter(s1)
    s4 = g.filter(s2)
    s5 = g.synthesize(s1)
    assert s3.shape == (test_graph.N,)
    np.testing.assert_allclose(s3, s4)
    np.testing.assert_allclose(s3, s5)

    s1 = rng.uniform(size=(test_graph.N, 10, n_filters))
    s3 = g.filter(s1)
    s5 = g.synthesize(s1)
    assert s3.shape == (test_graph.N, 10)
    np.testing.assert_allclose(s3, s5)


def test_localize(test_graph):
    """Test filter localization."""
    g = filters.Heat(test_graph, 100)

    # Localize signal at node by filtering Kronecker delta.
    NODE = 10
    s1 = g.localize(NODE, method="exact")

    # Should be equal to a row / column of the filtering operator.
    gL = test_graph.U.dot(np.diag(g.evaluate(test_graph.e)[0]).dot(test_graph.U.T))
    s2 = np.sqrt(test_graph.N) * gL[NODE, :]
    np.testing.assert_allclose(s1, s2)

    # That is actually a row / column of the analysis operator.
    F = g.compute_frame(method="exact")
    np.testing.assert_allclose(F, gL)


def test_frame_bounds(test_graph):
    """Test frame bounds estimation."""
    # Not a frame, it as a null-space.
    g = filters.Rectangular(test_graph)
    A, B = g.estimate_frame_bounds()
    assert A == 0
    assert B == 1
    # Identity is tight.
    g = filters.Filter(test_graph, lambda x: np.full_like(x, 2))
    A, B = g.estimate_frame_bounds()
    assert A == 4
    assert B == 4


def test_frame(test_graph):
    """Test that the frame is a stack of functions of the Laplacian."""
    g = filters.Heat(test_graph, scale=[8, 9])
    gL1 = g.compute_frame(method="exact")
    gL2 = g.compute_frame(method="chebyshev", order=30)

    def get_frame(freq_response):
        return test_graph.U.dot(np.diag(freq_response).dot(test_graph.U.T))

    gL = np.concatenate([get_frame(gl) for gl in g.evaluate(test_graph.e)])
    np.testing.assert_allclose(gL1, gL)
    np.testing.assert_allclose(gL2, gL, atol=1e-10)


def test_complement(test_graph, frame_bound=2.5):
    """Test that any filter bank becomes tight upon addition of their complement."""
    g = filters.MexicanHat(test_graph)
    g += g.complement(frame_bound)
    A, B = g.estimate_frame_bounds()
    np.testing.assert_allclose(A, frame_bound)
    np.testing.assert_allclose(B, frame_bound)


def test_inverse(test_graph, test_signal, caplog, frame_bound=3):
    """Test that the frame is the pseudo-inverse of the original frame."""
    g = filters.Heat(test_graph, scale=[2, 3, 4])
    h = g.inverse()
    Ag, Bg = g.estimate_frame_bounds()
    Ah, Bh = h.estimate_frame_bounds()
    np.testing.assert_allclose(Ag * Bh, 1)
    np.testing.assert_allclose(Bg * Ah, 1)
    gL = g.compute_frame(method="exact")
    hL = h.compute_frame(method="exact")
    Id = np.identity(test_graph.N)
    np.testing.assert_allclose(hL.T.dot(gL), Id, atol=1e-10)
    pinv = np.linalg.inv(gL.T.dot(gL)).dot(gL.T)
    np.testing.assert_allclose(pinv, hL.T, atol=1e-10)
    # The reconstruction is exact for any frame (lower bound A > 0).
    y = g.filter(test_signal, method="exact")
    z = h.filter(y, method="exact")
    np.testing.assert_allclose(z, test_signal)
    # Not invertible if not a frame.
    g = filters.Expwin(test_graph)
    with caplog.at_level("WARNING"):
        h = g.inverse()
        h.evaluate(test_graph.e)
    # Check that a warning was logged
    assert any("not invertible" in record.message for record in caplog.records)
    # If the frame is tight, inverse is h=g/A.
    g += g.complement(frame_bound)
    h = g.inverse()
    he = g(test_graph.e) / frame_bound
    np.testing.assert_allclose(h(test_graph.e), he, atol=1e-10)


def test_custom_filter(test_graph, test_signal):
    """Test custom filter creation."""

    def kernel(x):
        return x / (1.0 + x)

    f = filters.Filter(test_graph, kernels=kernel)
    assert f.Nf == 1
    assert f._kernels[0] is kernel
    _test_filter_methods(test_graph, test_signal, f, tight=False)


def test_abspline(test_graph, test_signal):
    """Test Abspline filter."""
    f = filters.Abspline(test_graph, Nf=4)
    _test_filter_methods(test_graph, test_signal, f, tight=False)


def test_gabor(test_graph, test_signal):
    """Test Gabor filter."""
    f = filters.Rectangular(test_graph, None, 0.1)
    f = filters.Gabor(test_graph, f)
    _test_filter_methods(test_graph, test_signal, f, tight=False, check=False)

    with pytest.raises(ValueError):
        filters.Gabor(graphs.Sensor(), f)
    f = filters.Regular(test_graph)
    with pytest.raises(ValueError):
        filters.Gabor(test_graph, f)


def test_modulation(test_graph, test_signal):
    """Test Modulation filter."""
    f = filters.Rectangular(test_graph, None, 0.1)
    # TODO: synthesis doesn't work yet.
    # f = filters.Modulation(test_graph, f, modulation_first=False)
    # _test_filter_methods(test_graph, test_signal, f, tight=False, check=False)
    f = filters.Modulation(test_graph, f, modulation_first=True)
    _test_filter_methods(test_graph, test_signal, f, tight=False, check=False)

    with pytest.raises(ValueError):
        filters.Modulation(graphs.Sensor(), f)
    f = filters.Regular(test_graph)
    with pytest.raises(ValueError):
        filters.Modulation(test_graph, f)


def test_modulation_gabor(test_graph, test_signal):
    """Test that both Modulation and Gabor should be equivalent
    for deltas centered at the eigenvalues.
    """
    f = filters.Rectangular(test_graph, 0, 0)
    f1 = filters.Modulation(test_graph, f, modulation_first=True)
    f2 = filters.Gabor(test_graph, f)
    s1 = f1.filter(test_signal)
    s2 = f2.filter(test_signal)
    np.testing.assert_allclose(abs(s1), abs(s2), atol=1e-5)


def test_halfcosine(test_graph, test_signal):
    """Test HalfCosine filter."""
    f = filters.HalfCosine(test_graph, Nf=4)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_itersine(test_graph, test_signal):
    """Test Itersine filter."""
    f = filters.Itersine(test_graph, Nf=4)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_mexicanhat(test_graph, test_signal):
    """Test MexicanHat filter."""
    f = filters.MexicanHat(test_graph, Nf=5, normalize=False)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.MexicanHat(test_graph, Nf=4, normalize=True)
    _test_filter_methods(test_graph, test_signal, f, tight=False)


def test_meyer(test_graph, test_signal):
    """Test Meyer filter."""
    f = filters.Meyer(test_graph, Nf=4)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_simpletf(test_graph, test_signal):
    """Test SimpleTight filter."""
    f = filters.SimpleTight(test_graph, Nf=4)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_regular(test_graph, test_signal):
    """Test Regular filter."""
    f = filters.Regular(test_graph)
    _test_filter_methods(test_graph, test_signal, f, tight=True)
    f = filters.Regular(test_graph, degree=5)
    _test_filter_methods(test_graph, test_signal, f, tight=True)
    f = filters.Regular(test_graph, degree=0)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_held(test_graph, test_signal):
    """Test Held filter."""
    f = filters.Held(test_graph)
    _test_filter_methods(test_graph, test_signal, f, tight=True)
    f = filters.Held(test_graph, a=0.25)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_simoncelli(test_graph, test_signal):
    """Test Simoncelli filter."""
    f = filters.Simoncelli(test_graph)
    _test_filter_methods(test_graph, test_signal, f, tight=True)
    f = filters.Simoncelli(test_graph, a=0.25)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_papadakis(test_graph, test_signal):
    """Test Papadakis filter."""
    f = filters.Papadakis(test_graph)
    _test_filter_methods(test_graph, test_signal, f, tight=True)
    f = filters.Papadakis(test_graph, a=0.25)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_heat(test_graph, test_signal):
    """Test Heat filter."""
    f = filters.Heat(test_graph, normalize=False, scale=10)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Heat(test_graph, normalize=False, scale=np.array([5, 10]))
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Heat(test_graph, normalize=True, scale=10)
    np.testing.assert_allclose(np.linalg.norm(f.evaluate(test_graph.e)), 1)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Heat(test_graph, normalize=True, scale=[5, 10])
    np.testing.assert_allclose(np.linalg.norm(f.evaluate(test_graph.e)[0]), 1)
    np.testing.assert_allclose(np.linalg.norm(f.evaluate(test_graph.e)[1]), 1)
    _test_filter_methods(test_graph, test_signal, f, tight=False)


def test_wave(test_graph, test_signal):
    """Test Wave filter."""
    f = filters.Wave(test_graph)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Wave(test_graph, time=1)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Wave(test_graph, time=[1, 2, 3])
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Wave(test_graph, speed=[1])
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Wave(test_graph, speed=[0.5, 1, 1.5])
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Wave(test_graph, time=[1, 2], speed=[1, 1.5])
    _test_filter_methods(test_graph, test_signal, f, tight=False)

    # Sequences of differing lengths.
    with pytest.raises(ValueError):
        filters.Wave(test_graph, time=[1, 2, 3], speed=[0, 1])
    # Invalid speed.
    with pytest.raises(ValueError):
        filters.Wave(test_graph, speed=2)


def test_expwin(test_graph, test_signal):
    """Test Expwin filter."""
    f = filters.Expwin(test_graph)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Expwin(test_graph, band_min=None, band_max=0.8)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Expwin(test_graph, band_min=0.1, band_max=None)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Expwin(test_graph, band_min=0.1, band_max=0.7)
    _test_filter_methods(test_graph, test_signal, f, tight=False)
    f = filters.Expwin(test_graph, band_min=None, band_max=None)
    _test_filter_methods(test_graph, test_signal, f, tight=True)


def test_rectangular(test_graph, test_signal):
    """Test Rectangular filter."""
    f = filters.Rectangular(test_graph)
    _test_filter_methods(test_graph, test_signal, f, tight=False, check=False)
    f = filters.Rectangular(test_graph, band_min=None, band_max=0.8)
    _test_filter_methods(test_graph, test_signal, f, tight=False, check=False)
    f = filters.Rectangular(test_graph, band_min=0.1, band_max=None)
    _test_filter_methods(test_graph, test_signal, f, tight=False, check=False)
    f = filters.Rectangular(test_graph, band_min=0.1, band_max=0.7)
    _test_filter_methods(test_graph, test_signal, f, tight=False, check=False)
    f = filters.Rectangular(test_graph, band_min=None, band_max=None)
    _test_filter_methods(test_graph, test_signal, f, tight=True, check=True)


def test_approximations(test_graph, test_signal):
    """
    Test that the different methods for filter analysis, i.e. 'exact',
    'cheby', and 'lanczos', produce the same output.
    """
    # TODO: done in _test_filter_methods.

    f = filters.Heat(test_graph)
    c_exact = f.filter(test_signal, method="exact")
    c_cheby = f.filter(test_signal, method="chebyshev")

    np.testing.assert_allclose(c_exact, c_cheby)

    with pytest.raises(ValueError):
        f.filter(test_signal, method="lanczos")
