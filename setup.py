#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name='PyGSP',
    version='0.5.1',
    description='Graph Signal Processing in Python',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='EPFL LTS2',
    url='https://github.com/epfl-lts2/pygsp',
    project_urls={
        'Documentation': 'https://pygsp.readthedocs.io',
        'Download': 'https://pypi.org/project/PyGSP',
        'Source Code': 'https://github.com/epfl-lts2/pygsp',
        'Bug Tracker': 'https://github.com/epfl-lts2/pygsp/issues',
        'Try It Online': 'https://mybinder.org/v2/gh/epfl-lts2/pygsp/master?filepath=playground.ipynb',
    },
    packages=[
        'pygsp',
        'pygsp.graphs',
        'pygsp.graphs.nngraphs',
        'pygsp.filters',
        'pygsp.tests',
    ],
    package_data={'pygsp': ['data/pointclouds/*.mat']},
    test_suite='pygsp.tests.suite',
    install_requires=[
        'numpy',
        'scipy',
    ],
    extras_require={
        # Optional dependencies for development. Some bring additional
        # functionalities, others are for testing, documentation, or packaging.
        'dev': [
            # Import and export.
            'networkx',
            # 'graph-tool', cannot be installed by pip
            # Construct patch graphs from images.
            'scikit-image',
            # Approximate nearest neighbors for kNN graphs.
            'pyflann3',
            # Convex optimization on graph.
            'pyunlocbox',
            # Plot graphs, signals, and filters.
            'matplotlib',
            # Interactive graph visualization.
            'pyqtgraph',
            'PyOpenGL',
            'PyQt5',
            # Run the tests.
            'flake8',
            'coverage',
            'coveralls',
            # Build the documentation.
            'sphinx',
            'numpydoc',
            'sphinxcontrib-bibtex',
            'sphinx-gallery',
            'memory_profiler',
            'sphinx-rtd-theme',
            'sphinx-copybutton',
            # Build and upload packages.
            'wheel',
            'twine',
        ],
    },
    license="BSD",
    keywords='graph signal processing',
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
