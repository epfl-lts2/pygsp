#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup


setup(
    name='PyGSP',
    version='0.4.2',
    description='Graph Signal Processing in Python',
    long_description=open('README.rst').read(),
    author='EPFL LTS2',
    url='https://github.com/epfl-lts2/pygsp',
    packages=['pygsp',
              'pygsp.graphs',
              'pygsp.graphs.nngraphs',
              'pygsp.filters',
              'pygsp.tests'],
    package_data={'pygsp': ['data/pointclouds/*.mat']},
    test_suite='pygsp.tests.test_all.suite',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pyqtgraph',
        'PyQt5' if sys.version_info >= (3, 5) else 'PySide',
        'pyopengl',
        'scikit-image',
        'pyflann' if sys.version_info.major == 2 else 'pyflann3'],
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
