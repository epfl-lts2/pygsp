============
Contributing
============

Contributions are welcome, and they are greatly appreciated! The development of
this package takes place on `GitHub <https://github.com/epfl-lts2/pygsp>`_.
Issues, bugs, and feature requests should be reported `there
<https://github.com/epfl-lts2/pygsp/issues>`_.
Code and documentation can be improved by submitting a `pull request
<https://github.com/epfl-lts2/pygsp/pulls>`_. Please add documentation and
tests for any new code.

The package can be set up (ideally in a fresh virtual environment) for local
development with the following::

    $ git clone https://github.com/epfl-lts2/pygsp.git
    $ cd pygsp
    $ make install

The ``make install`` command (which runs ``uv sync --all-extras``) ensures that
all dependencies required for development (to run the test suite and build the
documentation) are installed. Only `graph-tool <https://graph-tool.skewed.de>`_
will be missing: install it manually as it cannot be installed by uv.

You can improve or add functionality in the ``pygsp`` folder, along with
corresponding unit tests in ``pygsp/tests/test_*.py`` (with reasonable
coverage).
If you have a nice example to demonstrate the use of the introduced
functionality, please consider adding a tutorial in ``doc/tutorials`` or a
short example in ``examples``.

Update ``README.rst`` and ``CHANGELOG.rst`` if applicable.

After making any change, please check the style, run the tests, and build the
documentation with the following (enforced by GitHub Actions)::

    $ make lint
    $ make test
    $ make doc

Check the generated coverage report at ``htmlcov/index.html`` to make sure the
tests reasonably cover the changes you've introduced.

To iterate faster, you can partially run the test suite, at various degrees of
granularity, as follows::

   $ python -m pytest pygsp/tests/test_docstrings.py
   $ python -m pytest pygsp/tests/test_graphs.py::TestGraphs
   $ python -m pytest pygsp/tests/test_graphs.py::test_save_load

Making a release
----------------

#. Update the version number and release date in ``setup.py``,
   ``pygsp/__init__.py`` and ``CHANGELOG.rst``.
#. Create a git tag with ``git tag -a v0.5.0 -m "PyGSP v0.5.0"``.
#. Push the tag to GitHub with ``git push github v0.5.0``. The tag should now
   appear in the releases and tags tab.
#. `Create a release <https://github.com/epfl-lts2/pygsp/releases/new>`_ on
   GitHub and select the created tag. A DOI should then be issued by Zenodo.
#. Go on Zenodo and fix the metadata if necessary.
#. Build the distribution with ``make dist`` and check that the
   ``dist/PyGSP-0.5.0.tar.gz`` source archive contains all required files. The
   binary wheel should be found as ``dist/PyGSP-0.5.0-py2.py3-none-any.whl``.
#. Test the upload and installation process::

    $ uv publish --publish-url https://test.pypi.org/legacy/ dist/*
    $ pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pygsp

   Log in as the LTS2 user.
#. Build and upload the distribution to the real PyPI with ``make release``.
#. Update the conda feedstock (at least the version number and sha256 in
   ``recipe/meta.yaml``) by sending a PR to
   `conda-forge <https://github.com/conda-forge/pygsp-feedstock>`_.

Repository organization
-----------------------

::

  LICENSE.txt         Project license
  *.rst               Important documentation
  Makefile            Targets for make
  setup.py            Meta information about package (published on PyPI)
  .gitignore          Files ignored by the git revision control system
  .github/workflows/  Defines testing on GitHub Actions continuous integration

  pygsp/              Contains the modules (the actual toolbox implementation)
   __init.py__        Load modules at package import
   *.py               One file per module

  pygsp/tests/        Contains the test suites (will be distributed to end user)
   __init.py__        Load modules at package import
   test_*.py          One test suite per module
   test_docstrings.py Test the examples in the docstrings (reference doc)
   test_tutorials.py  Test the tutorials in doc/tutorials
   test_all.py        Launch all the tests (docstrings, tutorials, modules)

  doc/                Package documentation
   conf.py            Sphinx configuration
   index.rst          Documentation entry page
   *.rst              Include doc files from root directory

  doc/reference/      Reference documentation
   index.rst          Reference entry page
   *.rst              Only directives, the actual doc is alongside the code

  doc/tutorials/
   index.rst          Tutorials entry page
   *.rst              One file per tutorial
