"""pytest discovery shim.

Putting a ``conftest.py`` at the repo root means pytest auto-adds this
directory to ``sys.path``, so ``import analysis`` etc. just work without
having to install the package.
"""
