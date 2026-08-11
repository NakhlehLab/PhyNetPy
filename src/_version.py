"""Single source of truth for the PhyNetPy version.

``pyproject.toml`` reads ``__version__`` from this module statically (see
``[tool.setuptools.dynamic]``), and ``phynetpy/__init__.py`` re-exports it as
``phynetpy.__version__``.  Keep this file free of imports so that setuptools
can parse it without importing the package.

``deploy.py`` rewrites the literal below when bumping a release.
"""

__version__ = "0.6.0"
