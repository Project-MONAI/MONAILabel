# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../"))
import monailabel

# -- Project information -----------------------------------------------------

project = "MONAI Label"
copyright = "2020 - 2021 MONAI Consortium"
author = "MONAI Label Contributors"

# The full version, including alpha/beta/rc tags
short_version = monailabel.__version__.split("+")[0]
release = short_version
version = short_version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Add napoleon to the extensions list
# source_parsers = {'.md': CommonMarkParser}

templates_path = ["templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

extensions = [
    "recommonmark",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.exceltable",
]

autoclass_content = "both"
add_module_names = False
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": True,
    "display_version": True,
    "navigation_depth": 5,
    "sticky_navigation": True,  # Set to False to disable the sticky nav while scrolling.
    # 'logo_only': True,  # if we have a html_logo below, this shows /only/ the logo with no title text
}

html_scaled_image_link = False
html_show_sourcelink = True
html_favicon = "../images/favicon.ico"
html_logo = "../images/MONAI-logo-color.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../_static"]
html_css_files = ["custom.css"]
