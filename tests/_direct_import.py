"""Load individual scene/ modules without importing scene/__init__.py.

scene/__init__.py pulls in dataset readers, plyfile, and CUDA extensions;
the pure-Python control/scheduling modules under test have no need for any
of that, so tests load them directly by file path.
"""

import importlib.util
import os
import sys


def load_scene_module(name: str):
    """Load scene/<name>.py as a standalone module and return it."""
    path = os.path.join(os.path.dirname(__file__), "..", "scene", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register so dataclasses/pickling and intra-module imports resolve.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
