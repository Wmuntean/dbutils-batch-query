"""
=======================================================
Pytest Configuration
=======================================================

This module provides fixtures and configuration for pytest.

.. currentmodule:: tests.conftest

Fixtures
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    package_import
"""

import importlib
import os
import sys
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def package_import():
    """
    Import the package from its installed location or development directory.
    
    Returns
    -------
    module
        The imported package module.
    
    Notes
    -----
    This fixture ensures the package is properly imported for testing,
    whether it's installed or being run from source.
    """
    # Add the parent directory to sys.path
    package_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(package_dir))
    
    package_name = "dbutils_batch_query"
    
    try:
        # Try to import the package
        package = importlib.import_module(package_name)
        yield package
    finally:
        # Clean up sys.path after the test session
        if str(package_dir) in sys.path:
            sys.path.remove(str(package_dir))