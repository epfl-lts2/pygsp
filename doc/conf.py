#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygsp

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode',
              'sphinx.ext.autosummary', 'sphinx.ext.mathjax',
              'sphinx.ext.inheritance_diagram', 'sphinxcontrib.bibtex']

extensions.append('numpydoc')
numpydoc_show_class_members = False

exclude_patterns = ['_build']
source_suffix = '.rst'
master_doc = 'index'

project = 'PyGSP'
version = pygsp.__version__
release = pygsp.__version__
copyright = 'EPFL LTS2'

pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 2,
}
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}
latex_documents = [
    ('index', 'pygsp.tex', 'PyGSP documentation',
     'EPFL LTS2', 'manual'),
]
