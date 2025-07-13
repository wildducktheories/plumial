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
    d = D(133)

    assert d.n() == 7  # bit_length() - 1 = 8 - 1 = 7
    assert d.o() == 2  # bit_count() - 1 = 3 - 1 = 2
    assert d.e() == 5  # n - o = 7 - 2 = 5
    assert d.q() == 2  # e // o = 5 // 2 = 2
    assert d.r() == 1  # e % o = 5 % 2 = 1


def test_d_symbolic_expression():
    """Test symbolic expression generation."""
    d = D(21)  # p=21 has n=4, o=2, e=2

    expr = d.as_expr()
    expected = h**2 - g**2

    assert expr == expected
    assert str(d) == str(expected)


def test_d_evaluation():
    """Test polynomial evaluation with numeric values."""
    d = D(21)  # h^2 - g^2

    # Symbolic evaluation
    symbolic = d.d()
    assert symbolic == h**2 - g**2

    # Full evaluation using modern encoding
    numeric = d.encode(g=3, h=2).d()
    assert numeric == 4 - 9  # -5


def test_d_gcd():
    """Test GCD computation."""
    d = D(21)  # e=2, o=2

    # Symbolic GCD
    symbolic_gcd = d.gcd()
    assert str(symbolic_gcd) == "gcd(4, g**2)"

    # Numeric GCD
    numeric_gcd = d.gcd(g=2)
    assert numeric_gcd == math.gcd(4, 4)  # gcd(2^2, 2^2) = 4


def test_d_g_vector():
    """Test g-vector generation."""
    d = D(293)  # o=3

    # Symbolic vector
    vec = d.g_vector()
    expected = sy.Matrix([1, g, g**2])
    assert vec == expected

    # Numeric vector
    vec_numeric = d.g_vector(g=3)
    expected_numeric = sy.Matrix([1, 3, 9])
    assert vec_numeric == expected_numeric

    # Matrix form
    mat = d.g_vector(matrix=True)
    expected_mat = sy.diag(1, g, g**2)
    assert mat == expected_mat


def test_d_factorization():
    """Test polynomial factorization."""
    d = D(21)  # h^2 - g^2

    factored = d.factor()
    # h^2 - g^2 = (h - g)(h + g)
    expected = (h - g) * (h + g)
    assert sy.expand(factored) == sy.expand(expected)


def test_d_caching():
    """Test D instance caching."""
    clear_d_cache()

    d1 = D(21)
    d2 = D(21)

    # Should be the same cached instance
    assert d1 is d2

    # Cache should show hits
    info = d_cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_d_edge_cases():
    """Test edge cases for D class."""
    # Case where o = 0
    d_zero_o = D(8)
    assert d_zero_o.o() == 0
    assert d_zero_o.e() == 3
    assert d_zero_o.q() == 0
    assert d_zero_o.r() == 0
    assert d_zero_o.as_expr() == h**3 - 1  # g^0 = 1

    # Case where e = 0
    d_zero_e = D(7)
    assert d_zero_e.o() == 2
    assert d_zero_e.e() == 0
    assert d_zero_e.as_expr() == 1 - g**2  # h^0 = 1


def test_d_equality_and_hashing():
    """Test equality and hashing."""
    d1 = D(21)
    d2 = D(21)
    d3 = D(23)

    assert d1 == d2
    assert d1 != d3
    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)


def test_d_representation():
    """Test string representations."""
    d = D(21)

    # SymPy may order terms differently, so just check key components
    expr_str = str(d)
    assert "h**2" in expr_str
    assert "g**2" in expr_str
    assert "D(n=4, o=2, e=2)" in repr(d)
    assert "h**2" in repr(d) and "g**2" in repr(d)


def test_d_integration_with_p():
    """Test D integration with P class."""
    from plumial import P

    p = P(133)
    d = p.D()

    # Should be the same as creating D directly
    d_direct = D(133)
    assert d.n() == d_direct.n()
    assert d.o() == d_direct.o()
    assert d.e() == d_direct.e()

    # Test polynomial evaluation through P
    assert p.d() == d.as_expr()
    assert p.encode(g=3, h=2).d() == d.encode(g=3, h=2).d()
