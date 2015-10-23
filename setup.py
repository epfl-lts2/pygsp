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
    version = '0.2.1',
    description = 'The official Graph Signal Processing Toolbox',
    long_description = open('README.rst').read(),
    author = 'Basile Châtillon, Alexandre Lafaye , Lionel Martin, Nicolas Rod (EPFL LTS2)',
    author_email = 'basile.chatillon@epfl.ch, alexandre.lafaye@epfl.ch, lionel.martin@epfl.ch, nicolas.rod@epfl.ch',
    url = 'https://github.com/epfl-lts2/pygsp',
    packages = ['pygsp', 'pygsp.filters', 'pygsp.graphs', 'pygsp.graphs.nngraphs', 'pygsp.operators', 'pygsp.pointsclouds', 'pygsp.tests'],
    package_data = {'pygsp': ['pointsclouds.misc/*']},
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
)
