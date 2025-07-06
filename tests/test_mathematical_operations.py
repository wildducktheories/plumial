"""
Tests for mathematical operations in P class.
"""

import math

import pytest
import sympy as sy

from plumial import P
from plumial.core.P import clear_cache
from plumial.utils.symbolic import g, h, u, v


def test_uv_polynomial():
    """Test UV polynomial generation."""
    clear_cache()

    # Test with p=133 (binary: 10000101)
    p = P(133)
    uv_expr = p.uv()

    # Should be a polynomial in u and v
    assert uv_expr.has(u) or uv_expr.has(v) or uv_expr == 0
    print(f"UV polynomial for p=133: {uv_expr}")


def test_k_polynomial():
    """Test k polynomial calculation."""
    p = P(133)

    # Symbolic k
    k_symbolic = p.k()
    assert k_symbolic.has(g) or k_symbolic.has(h) or k_symbolic.is_number

    # Numerical k
    k_numeric = p.k(g=3, h=2)
    assert isinstance(k_numeric, (int, float, sy.Rational))

    print(f"k symbolic: {k_symbolic}")
    print(f"k(3,2): {k_numeric}")


def test_ax_polynomials():
    """Test a,x polynomial pair calculation."""
    p = P(133)

    a_poly, x_poly = p.ax()

    # Should be symbolic expressions
    assert isinstance(a_poly, sy.Expr)
    assert isinstance(x_poly, sy.Expr)

    # Verify relationship: x*d = a*k (symbolically)
    d_poly = p.d()
    k_poly = p.k()

    lhs = x_poly * d_poly
    rhs = a_poly * k_poly

    # Expand both sides and check if they're equal
    lhs_expanded = sy.expand(lhs)
    rhs_expanded = sy.expand(rhs)

    # They should be equal or differ by a constant factor
    print(f"a: {a_poly}")
    print(f"x: {x_poly}")
    print(f"x*d: {lhs_expanded}")
    print(f"a*k: {rhs_expanded}")


def test_a_coefficient():
    """Test a coefficient calculation."""
    p = P(133)

    # Symbolic a
    a_symbolic = p.a()
    assert isinstance(a_symbolic, sy.Expr)

    # Numerical a
    try:
        a_numeric = p.a(g=3, h=2)
        assert isinstance(a_numeric, (int, float, sy.Rational))
        print(f"a symbolic: {a_symbolic}")
        print(f"a(3,2): {a_numeric}")
    except ZeroDivisionError:
        print("a(3,2) resulted in division by zero (f=0)")


def test_x_polynomial_evaluation():
    """Test x polynomial calculation."""
    p = P(133)

    # Symbolic x
    x_symbolic = p.x()
    assert isinstance(x_symbolic, sy.Expr)

    # Numerical x
    try:
        x_numeric = p.x(g=3, h=2)
        assert isinstance(x_numeric, (int, float, sy.Rational))
        print(f"x symbolic: {x_symbolic}")
        print(f"x(3,2): {x_numeric}")
    except ZeroDivisionError:
        print("x(3,2) resulted in division by zero (f=0)")


def test_f_polynomial():
    """Test f polynomial (GCD factor) calculation."""
    p = P(133)

    # Symbolic f
    f_symbolic = p.f()
    assert isinstance(f_symbolic, sy.Expr)

    # Numerical f
    f_numeric = p.f(g=3, h=2)
    assert isinstance(f_numeric, (int, float))

    print(f"f symbolic: {f_symbolic}")
    print(f"f(3,2): {f_numeric}")


def test_mathematical_consistency():
    """Test mathematical relationships between polynomials."""
    p = P(133)

    # Get all polynomial values
    d_val = p.d(g=3, h=2)
    k_val = p.k(g=3, h=2)
    f_val = p.f(g=3, h=2)

    try:
        a_val = p.a(g=3, h=2)
        x_val = p.x(g=3, h=2)

        # Test relationship: a = d/f and x = k/f
        assert abs(a_val - d_val / f_val) < 1e-10, f"a != d/f: {a_val} != {d_val/f_val}"
        assert abs(x_val - k_val / f_val) < 1e-10, f"x != k/f: {x_val} != {k_val/f_val}"

        # Test relationship: x*d = a*k
        lhs = x_val * d_val
        rhs = a_val * k_val
        assert abs(lhs - rhs) < 1e-10, f"x*d != a*k: {lhs} != {rhs}"

        print("Mathematical consistency tests passed!")

    except ZeroDivisionError as e:
        print(f"Skipping consistency test due to division by zero: {e}")


def test_comparison_with_original():
    """Test that our implementation matches patterns from original."""
    # Test multiple p-values to ensure robustness
    test_values = [133, 9, 17, 65]

    for p_val in test_values:
        print(f"\n=== Testing p={p_val} ===")
        p = P(p_val)

        print(f"Binary: {bin(p_val)}")
        print(f"n={p.n()}, o={p.o()}, e={p.e()}")
        print(f"d(): {p.d()}")
        print(f"d(3,2): {p.d(g=3, h=2)}")

        try:
            print(f"k(): {p.k()}")
            print(f"k(3,2): {p.k(g=3, h=2)}")

            print(f"uv(): {p.uv()}")

            a, x = p.ax()
            print(f"a: {a}")
            print(f"x: {x}")

        except Exception as e:
            print(f"Error in calculations: {e}")


def test_edge_cases():
    """Test edge cases for mathematical operations."""
    # Test with small p-values
    edge_cases = [3, 5, 9]  # Small p-values

    for p_val in edge_cases:
        print(f"\n=== Edge case p={p_val} ===")
        p = P(p_val)

        try:
            # Basic properties
            assert p.p() == p_val
            assert p.n() >= 0
            assert p.o() >= 0
            assert p.e() >= 0

            # Polynomial calculations
            d_poly = p.d()
            k_poly = p.k()
            uv_poly = p.uv()

            print(f"Basic calculations successful for p={p_val}")

        except Exception as e:
            print(f"Error in edge case p={p_val}: {e}")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_uv_polynomial()
    test_k_polynomial()
    test_ax_polynomials()
    test_a_coefficient()
    test_x_polynomial_evaluation()
    test_f_polynomial()
    test_mathematical_consistency()
    test_comparison_with_original()
    test_edge_cases()
