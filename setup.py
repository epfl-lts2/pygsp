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
    name='PyGSP',
    version='0.4.0',
    description='The official Graph Signal Processing Toolbox',
    long_description=open('README.rst').read(),
    author='Alexandre Lafaye, Basile Châtillon, Lionel Martin, Nicolas Rod (EPFL LTS2)',
    author_email='alexandre.lafaye@epfl.ch, basile.chatillon@epfl.ch, lionel.martin@epfl.ch, nicolas.rod@epfl.ch',
    url='https://github.com/epfl-lts2/',
    packages=['pygsp', 'pygsp.filters', 'pygsp.graphs', 'pygsp.graphs.nngraphs', 'pygsp.operators',
              'pygsp.pointsclouds', 'pygsp.tests'],
    package_data={'pygsp.pointsclouds': ['misc/*.mat']},
    test_suite='pygsp.tests.test_all.suite',
    dependency_links = ['https://github.com/pyqtgraph/pyqtgraph@develop#egg=pyqtgraph-dev'],
    install_requires=['numpy', 'scipy', 'pyopengl', 'pyqtgraph==dev',
                      'matplotlib==1.4.3' if sys.version_info.major == 3 and sys.version_info.minor < 4 else 'matplotlib',
                      'PyQt5' if sys.version_info.major == 3 and sys.version_info.minor == 5 else 'PySide'],
    license="BSD",
    keywords='graph signal processing toolbox filters pointclouds',
    platforms='any',
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Environment :: Console',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
