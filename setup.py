#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

setup(
    name = 'pygsp',
    version = '0.0.1',
    description = 'The official Graph Processing Toolbox',
    long_description = open('README.rst').read(),
    author = 'Alexandre Lafaye (EPFL LTS2), Basile Châtillon (EPFL LTS2), Nicolas Rod (EPFL LTS2)',
    author_email = 'alexandre.lafaye@epfl.ch, basile.chatillon@epfl.ch, nicolas.rod@epfl.ch',
    url = 'https://github.com/epfl-lts2/',
    packages = ['pygsp', 'pygsp.tests'],
    package_data = {'pygsp': ['misc/*']},
    test_suite = 'pygsp.tests.test_all.suite',
    setup_requires = ['numpy',],
    install_requires = ['numpy', 'scipy'],
    requires = ['numpy'],
    license = "BSD",
    keywords = '',
    platforms = 'any',
    classifiers = [
        'Topic :: Scientific/Engineering :: Mathematics',
        'Environment :: Console',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
)
