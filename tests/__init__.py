"""Test package.

Exists so that `tests` is a regular package rather than a namespace one:
some third-party distributions ship a top-level `tests/` into site-packages,
which would otherwise shadow this directory and break `from tests.conftest
import ...`.
"""
