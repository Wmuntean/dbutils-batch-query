"""
=======================================================
Test Package Import and Version
=======================================================

This module tests basic functionality of the dbutils_batch_query package.

.. currentmodule:: tests.test_package

Tests
=====

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    test_import
    test_version
"""

import re


def test_import(package_import):
    """
    Test that the package can be imported.
    
    Parameters
    ----------
    package_import : module
        The imported package module from the conftest fixture.
    """
    assert package_import is not None


def test_version(package_import):
    """
    Test that the package has a valid version number.
    
    Parameters
    ----------
    package_import : module
        The imported package module from the conftest fixture.
    """
    version = package_import.__version__
    assert version is not None
    assert isinstance(version, str)
    
    # Version should follow semantic versioning (e.g., 0.1.0)
    assert re.match(r'^\d+\.\d+\.\d+', version) is not None