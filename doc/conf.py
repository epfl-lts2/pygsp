# -*- coding: utf-8 -*-

import pygsp

extensions = ['sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx.ext.inheritance_diagram',
              'sphinxcontrib.bibtex']

extensions.append('sphinx.ext.autodoc')
autodoc_default_flags = ['members', 'undoc-members']
autodoc_member_order = 'groupwise'  # alphabetical, groupwise, bysource

extensions.append('matplotlib.sphinxext.plot_directive')
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_working_directory = '.'

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
