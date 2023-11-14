import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

from cappr import __version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CAPPr"
copyright = "2023, kddubey"
author = "kddubey"

version = __version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx_togglebutton",
]
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = [os.path.join("css", "custom.css")]
html_context = {
    "display_github": True,
    "github_user": "kddubey",
    "github_repo": "cappr",
    "github_version": "main",
    "doc_path": "/docs/source/",
}
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# see example: https://github.com/pydata/pydata-sphinx-theme/blob/185a37aa36820f77bffa4c87a772092e9e7cc380/docs/conf.py#L116C12-L116C12
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/kddubey/cappr",
            "icon": "fa-brands fa-github",
        },
    ]
}


def setup(self):
    # Use cappr.Example not cappr.example.Example
    from cappr import Example

    Example.__module__ = "cappr"


def linkcode_resolve(domain, info):
    import importlib
    import inspect

    code_url = "https://github.com/kddubey/cappr/blob/main"
    ## ty https://github.com/readthedocs/sphinx-autoapi/issues/202#issuecomment-907582382
    if domain != "py":
        return
    if not info["module"]:
        return

    mod = importlib.import_module(info["module"])

    if info["fullname"] == "Example":
        Example = getattr(mod, info["fullname"])
        Example.__module__ = "cappr._example"

    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    ## Unwrap the object to get the correct source file in case that is wrapped by a
    ## decorator
    obj = inspect.unwrap(obj)

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    if info["fullname"] == "Example":
        Example = getattr(mod, info["fullname"])
        Example.__module__ = "cappr"

    file = file.lstrip("../")
    return f"{code_url}/{file}#L{start}-L{end}"


# ty https://stackoverflow.com/q/67473396/18758987
autodoc_type_aliases = {
    "Sequence": "Sequence",
}
