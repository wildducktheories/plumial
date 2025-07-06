"""
Basic tests for the plumial package.
"""

import pytest

import plumial


def test_version():
    """Test that the package version is accessible."""
    assert hasattr(plumial, "__version__")
    assert isinstance(plumial.__version__, str)


def test_package_imports():
    """Test that the package can be imported without errors."""
    import plumial

    assert plumial is not None
