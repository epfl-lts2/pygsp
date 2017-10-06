#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name='PyGSP',
    version='0.4.2',
    description='Graph Signal Processing in Python',
    long_description=open('README.rst').read(),
    author='EPFL LTS2',
    url='https://github.com/epfl-lts2/pygsp',
    packages=[
        'pygsp',
        'pygsp.graphs',
        'pygsp.graphs.nngraphs',
        'pygsp.filters',
        'pygsp.tests',
    ],
    package_data={'pygsp': ['data/pointclouds/*.mat']},
    test_suite='pygsp.tests.test_all.suite',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pyqtgraph',
        # PyQt5 is only available on PyPI as wheels for Python 3.5 and up.
        'PyQt5; python_version >= "3.5"',
        # No source package for PyQt5 on PyPI, fall back to PySide.
        'PySide; python_version < "3.5"',
        'pyopengl',
        'scikit-image',
        'pyflann; python_version == "2.*"',
        'pyflann3; python_version == "3.*"',
    ],
    extras_require={
        'test': [
            'pyunlocbox',
            'flake8',
            'coverage',
            'coveralls',
        ],
        'doc': [
            'pyunlocbox',
            'sphinx',
            'numpydoc',
            'sphinxcontrib-bibtex',
            'sphinx-rtd-theme',
        ],
        'pkg': [
            'wheel',
            'twine',
        ],
    },
    license="BSD",
    keywords='graph signal processing',
    platforms='any',
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Environment :: Console',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
