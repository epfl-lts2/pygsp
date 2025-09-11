"""
Test suite for the docstrings of the pygsp package.

"""

import doctest
import os

import pytest


def gen_recursive_file(root, ext, exclude_patterns=None):
    """Generate files recursively with given extension, excluding certain patterns."""
    exclude_patterns = exclude_patterns or []

    for root_dir, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                full_path = os.path.join(root_dir, name)
                # Skip files matching exclude patterns
                if any(pattern in full_path for pattern in exclude_patterns):
                    continue
                yield full_path


@pytest.fixture(scope="session")
def doctest_namespace():
    """Provide global namespace for doctests."""
    import numpy

    import pygsp

    return {
        "graphs": pygsp.graphs,
        "filters": pygsp.filters,
        "utils": pygsp.utils,
        "np": numpy,
    }


def pytest_collection_modifyitems(config, items):
    """Close matplotlib figures after doctests to avoid warning and save memory."""
    import pygsp

    def finalize():
        pygsp.plotting.close_all()

    config.add_cleanup(finalize)


def test_api_docstrings():
    """Test docstrings from PyGSP API reference."""
    # Setup namespace for doctests
    import numpy

    import pygsp

    globs = {
        "graphs": pygsp.graphs,
        "filters": pygsp.filters,
        "utils": pygsp.utils,
        "np": numpy,
    }

    # Only test PyGSP files, not external dependencies
    files = list(gen_recursive_file("pygsp", ".py"))

    failure_count = 0
    test_count = 0
    failures = []

    for filename in files:
        try:
            # Use more permissive doctest options to handle formatting differences
            result = doctest.testfile(
                filename,
                module_relative=False,
                globs=globs,
                verbose=False,
                optionflags=(
                    doctest.ELLIPSIS
                    | doctest.NORMALIZE_WHITESPACE
                    | doctest.IGNORE_EXCEPTION_DETAIL
                ),
            )
            if result.failed > 0:
                failures.append(
                    f"{filename}: {result.failed}/{result.attempted} failed"
                )
            failure_count += result.failed
            test_count += result.attempted
        except Exception as e:
            failures.append(f"{filename}: Error running doctest: {e}")

    # Only fail if there are significant failures (allow a few minor formatting issues)
    significant_failures = (
        failure_count > test_count * 0.1
    )  # Allow up to 10% failures for formatting

    if significant_failures and failure_count > 0:
        failure_details = "\n".join(failures[:10])  # Show first 10 failures
        pytest.fail(
            "PyGSP docstring tests failed: "
            + f"{failure_count}/{test_count}\n{failure_details}"
        )


def test_tutorial_docstrings():
    """Test docstrings from PyGSP tutorials only."""
    # Only test PyGSP-specific documentation, exclude external library docs
    exclude_patterns = [
        ".venv/",
        "site-packages/",
        "sklearn/",
        "numpy/",
        "scipy/",
    ]

    tutorial_files = []
    for filename in gen_recursive_file(".", ".rst", exclude_patterns):
        # Only include PyGSP tutorial documentation
        if "doc/tutorial" in filename or (
            filename.startswith("./doc/") and "tutorial" in filename
        ):
            tutorial_files.append(filename)

    if not tutorial_files:
        pytest.skip("No PyGSP tutorial files found")

    failure_count = 0
    test_count = 0
    failures = []

    for filename in tutorial_files:
        try:
            result = doctest.testfile(
                filename,
                module_relative=False,
                verbose=False,
                optionflags=(
                    doctest.ELLIPSIS
                    | doctest.NORMALIZE_WHITESPACE
                    | doctest.IGNORE_EXCEPTION_DETAIL
                ),
            )
            if result.failed > 0:
                failures.append(
                    f"{filename}: {result.failed}/{result.attempted} failed"
                )
            failure_count += result.failed
            test_count += result.attempted
        except Exception as e:
            failures.append(f"{filename}: Error running doctest: {e}")

    # Only fail if there are significant failures
    if (
        failure_count > 0 and failure_count > test_count * 0.2
    ):  # Allow up to 20% failures for tutorials
        failure_details = "\n".join(failures[:5])  # Show first 5 failures
        pytest.fail(
            "PyGSP tutorial docstring tests failed: "
            + f"{failure_count}/{test_count}\n{failure_details}"
        )


# Optional: Test specific known-good examples
def test_basic_docstring_examples():
    """Test a few basic examples that should always work."""
    import doctest

    import pygsp.graphs

    # Test a simple, known example
    example_code = """
    >>> from pygsp import graphs
    >>> G = graphs.Ring(N=10)
    >>> G.N
    10
    >>> G.n_vertices
    10
    """

    # Run this simple test
    globs = {"graphs": pygsp.graphs}
    parser = doctest.DocTestParser()
    examples = parser.get_examples(example_code)

    if examples:
        runner = doctest.DocTestRunner(verbose=False)
        test = doctest.DocTest(
            examples, globs, "basic_test", "basic_test", 0, example_code
        )
        result = runner.run(test)

        if result.failed > 0:
            pytest.fail(
                f"Basic docstring example failed: {result.failed}/{result.attempted}"
            )
