.. _contributing:

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

The package can be set up (ideally in a virtual environment) for local
development with the following::

    $ git clone git@github.com:epfl-lts2/pygsp.git
    $ pip install -U -r pygsp/requirements.txt
    $ pip install -e pygsp

You can improve or add functionality in the ``pygsp`` folder, along with
corresponding unit tests in ``pygsp/tests/test_*.py`` (with reasonable
coverage) and documentation in ``doc/reference/*.rst``. If you have a nice
example to demonstrate the use of the introduced functionality, please consider
adding a tutorial in ``doc/tutorials``.

Do not forget to update ``README.rst`` and ``doc/history.rst`` with e.g. new
features. The version number needs to be updated in ``setup.py`` and
``pyunlocbox/__init__.py``.

After making any change, please check the style, run the tests, and build the
documentation with the following (enforced by Travis CI)::

    $ make lint
    $ make test
    $ make doc

Check the generated coverage report at ``htmlcov/index.html`` to make sure the
tests reasonably cover the changes you've introduced.
