"""
Tests for d-polynomial functionality.
"""

import math

import pytest
import sympy as sy

from plumial.core.D import D, clear_d_cache, d_cache_info
from plumial.utils.symbolic import g, h


def test_d_from_p_value():
    """Test D creation from p-value."""
    clear_d_cache()

    # Test with p=133 (binary: 10000101)
    d = D(2, 5)

    assert d.n() == 7  # bit_length() - 1 = 8 - 1 = 7
    assert d.o() == 2  # bit_count() - 1 = 3 - 1 = 2
    assert d.e() == 5  # n - o = 7 - 2 = 5
    # c() and r() return symbolic expressions when not encoded to a specific basis


def test_d_symbolic_expression():
    """Test symbolic expression generation."""
    d = D(2, 2)  # n=4, o=2, e=2

    expr = d._as_expr()
    expected = h**2 - g**2

    assert expr == expected
    assert str(d) == str(expected)


def test_d_evaluation():
    """Test polynomial evaluation with numeric values."""
    d = D(2, 2)  # h^2 - g^2

    # Symbolic evaluation
    symbolic = d.d()
    assert symbolic == h**2 - g**2

    # Full evaluation using modern encoding
    numeric = d.encode(g=3, h=2).d()
    assert numeric == 4 - 9  # -5






def test_d_caching():
    """Test D instance caching."""
    clear_d_cache()

    d1 = D(2, 2)
    d2 = D(2, 2)

    # Should be the same cached instance
    assert d1 is d2

    # Cache should show hits
    info = d_cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_d_edge_cases():
    """Test edge cases for D class."""
    # Case where o = 0
    d_zero_o = D(0, 3)
    assert d_zero_o.o() == 0
    assert d_zero_o.e() == 3
    assert d_zero_o._as_expr() == h**3 - 1  # g^0 = 1

    # Case where e = 0
    d_zero_e = D(2, 0)
    assert d_zero_e.o() == 2
    assert d_zero_e.e() == 0
    assert d_zero_e._as_expr() == 1 - g**2  # h^0 = 1


def test_d_equality_and_hashing():
    """Test equality and hashing."""
    d1 = D(2, 2)
    d2 = D(2, 2)
    d3 = D(3, 1)

    assert d1 == d2
    assert d1 != d3
    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)


def test_d_representation():
    """Test string representations."""
    d = D(2, 2)

    # SymPy may order terms differently, so just check key components
    expr_str = str(d)
    assert "h**2" in expr_str
    assert "g**2" in expr_str
    
    # repr should now be the same as the expression repr (no D(o=..., e=...) prefix)
    repr_str = repr(d)
    assert "h**2" in repr_str and "g**2" in repr_str
    
    # Test that D repr equals SymPy expression repr
    assert repr(d) == repr(d._as_expr())
    assert repr(d) == repr(d * 1)  # This is the key requirement


def test_d_integration_with_p():
    """Test D integration with P class."""
    from plumial import P

    p = P(133)
    d = p.D()

    # Should be the same as creating D directly
    d_direct = D(2, 5)
    assert d.n() == d_direct.n()
    assert d.o() == d_direct.o()
    assert d.e() == d_direct.e()

    # Test polynomial evaluation through P
    assert p.d() == d._as_expr()
    assert p.encode(g=3, h=2).d() == d.encode(g=3, h=2).d()
