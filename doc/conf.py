import pygsp

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
]

extensions.append("sphinx.ext.autodoc")
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "groupwise",  # alphabetical, groupwise, bysource
}

extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pyunlocbox": ("https://pyunlocbox.readthedocs.io/en/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "graph_tool": ("https://graph-tool.skewed.de/static/doc", None),
}

extensions.append("numpydoc")
numpydoc_show_class_members = False
numpydoc_use_plots = True  # Add the plot directive whenever mpl is imported.

extensions.append("matplotlib.sphinxext.plot_directive")
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_working_directory = "."
plot_rcparams = {"figure.figsize": (10, 4)}
plot_pre_code = """
import numpy as np
from pygsp import graphs, filters, utils, plotting
"""

extensions.append("sphinx_gallery.gen_gallery")
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
    "filename_pattern": "/",
    "reference_url": {"pygsp": None},
    "backreferences_dir": "backrefs",
    "doc_module": "pygsp",
    "show_memory": True,
}

extensions.append("sphinx_copybutton")
copybutton_prompt_text = ">>> "

extensions.append("sphinxcontrib.bibtex")
bibtex_bibfiles = ["references.bib"]

exclude_patterns = ["_build"]
source_suffix = ".rst"
master_doc = "index"

project = "PyGSP"
version = pygsp.__version__
release = pygsp.__version__
copyright = "EPFL LTS2"

pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
}
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
}
latex_documents = [
    ("index", "pygsp.tex", "PyGSP documentation", "EPFL LTS2", "manual"),
]
