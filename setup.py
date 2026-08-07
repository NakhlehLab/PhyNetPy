"""Build hook for PhyNetPy's Cython extensions.

All package metadata lives in ``pyproject.toml``; this file exists only to
compile the Cython modules, which cannot be declared declaratively.

``graph_core_cy`` backs every Network and is imported unconditionally, so a
build that skips or fails the compile step does not produce a usable package.
Compilation errors are therefore fatal rather than silently downgraded.
"""

import os

from setuptools import Extension, setup
from Cython.Build import cythonize

CYTHON_MODULES = [
    "graph_core_cy",   # NodeSet / EdgeSet adjacency maps
    "mpl_engine_cy",   # MPL triplet DP
    "gt_msc_cy",       # MSC / MSNC ancestral-configuration DP
    "seq_engine_cy",   # sequence-likelihood branch kernels
]

setup(
    ext_modules=cythonize(
        [
            Extension(
                f"phynetpy.cython.{name}",
                [f"src/cython/{name}.pyx"],
                extra_compile_args=[] if os.name == "nt" else ["-O3"],
            )
            for name in CYTHON_MODULES
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        quiet=True,
    )
)
