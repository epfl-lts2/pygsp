"""
Test suite for the utils module of the pygsp package.

"""

import numpy as np
import pytest
from scipy import sparse

from pygsp import utils


def test_symmetrize():
    """Test matrix symmetrization methods."""
    W = sparse.random(100, 100, random_state=42)
    for method in ["average", "maximum", "fill", "tril", "triu"]:
        # Test that the regular and sparse versions give the same result.
        W1 = utils.symmetrize(W, method=method)
        W2 = utils.symmetrize(W.toarray(), method=method)
        np.testing.assert_equal(W1.toarray(), W2)

    # Test that invalid method raises ValueError
    with pytest.raises(ValueError):
        utils.symmetrize(W, "sum")
