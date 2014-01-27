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

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='pygsp',
    version='0.1.0',
    description='Graph signal processing toolbox',
    long_description=readme + '\n\n' + history,
    author='LTS2 Graph Task Force',
    author_email='LTS2Graph@groupes.epfl.ch',
    url='https://github.com/epfl-lts2/pygsp',
    packages=[
        'pygsp',
    ],
    package_dir={'pygsp': 'pygsp'},
    include_package_data=True,
    install_requires=[
    ],
    license="GPL",
    zip_safe=False,
    keywords='pygsp',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPL v3 License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
)
