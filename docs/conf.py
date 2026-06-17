"""Sphinx configuration for pz-rail-svc documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "pz-rail-svc"
copyright = "2024, The LSST DESC PZ WG"
author = "The LSST DESC PZ WG"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "sphinx_click",
    "sphinx_tabs.tabs",
]

# AutoAPI configuration
autoapi_type = "python"
autoapi_dirs = ["../src/rail_svc"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_ignore = ["*/_version.py", "*/__pycache__/*"]
autoapi_keep_files = True

# Napoleon (numpy docstring support)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

html_static_path = ["_static"]

# General
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
