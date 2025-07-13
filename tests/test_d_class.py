"""
Simplified tests for the D class (d-polynomials).
"""

import pytest
import sympy as sy

from plumial.core.D import D, clear_d_cache, d_cache_info
from plumial.core import B
from plumial.utils.symbolic import g, h


def test_d_class_creation():
    """Test D class creation and basic properties."""
    clear_d_cache()

    # Test with D(2, 5) - equivalent to old p=133
    d_obj = D(2, 5)
    assert d_obj.o() == 2
    assert d_obj.e() == 5
    assert d_obj.n() == 7  # computed as o + e

    # Test with various o, e combinations
    test_cases = [(1, 0), (1, 1), (1, 2), (1, 3)]  # equivalent to old p-values [3, 5, 9, 17]
    for o_val, e_val in test_cases:
        d_obj = D(o_val, e_val)
        assert d_obj.o() == o_val
        assert d_obj.e() == e_val
        assert d_obj.n() == o_val + e_val


def test_d_caching():
    """Test LRU caching behavior."""
    clear_d_cache()

    # Create instances and check caching
    d1 = D(2, 5)
    d2 = D(2, 5)

    # Should be the same object due to caching
    assert d1 is d2

    # Check cache info
    info = d_cache_info()
    assert info.hits >= 1
    assert info.misses >= 1


def test_d_polynomial_evaluation():
    """Test d-polynomial evaluation."""
    d_obj = D(2, 5)

    # Symbolic evaluation
    d_symbolic = d_obj.d()
    assert isinstance(d_symbolic, sy.Expr)
    assert d_symbolic.has(g) or d_symbolic.has(h) or d_symbolic.is_number

    # Numerical evaluation
    d_numeric = d_obj.encode(B.Collatz).d()
    assert isinstance(d_numeric, (int, float, sy.Rational, sy.Integer, sy.Float))

    print(f"D(2, 5) symbolic: {d_symbolic}")
    print(f"D(2, 5) with Collatz basis: {d_numeric}")


def test_d_as_expr():
    """Test as_expr method."""
    d_obj = D(2, 5)
    expr = d_obj.as_expr()

    assert isinstance(expr, sy.Expr)
    assert expr.has(g) or expr.has(h)

    print(f"D(2, 5).as_expr(): {expr}")


def test_d_str_repr():
    """Test string representations."""
    d_obj = D(2, 5)

    str_repr = str(d_obj)
    repr_repr = repr(d_obj)

    assert isinstance(str_repr, str)
    assert isinstance(repr_repr, str)

    print(f"str(D(2, 5)): {str_repr}")
    print(f"repr(D(2, 5)): {repr_repr}")


def test_d_equality():
    """Test equality comparison."""
    d1 = D(2, 5)
    d2 = D(2, 5)
    d3 = D(1, 2)

    assert d1 == d2
    assert d1 != d3
    assert d1 != "not a D object"

    # Test hash consistency
    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)


def test_d_factory_patterns():
    """Test different D factory function calling patterns."""
    # All these should create the same D object: o=2, e=5
    d1 = D(2, 5)                    # Positional
    d2 = D(o=2, e=5)               # Named o, e
    d3 = D(o=2, n=7)               # Named o, n (compute e = n - o = 7 - 2 = 5)
    d4 = D(e=5, n=7)               # Named e, n (compute o = n - e = 7 - 5 = 2)
    d5 = D(o=2, e=5, n=7)          # All three (consistent)

    # All should be equal
    assert d1 == d2 == d3 == d4 == d5
    assert all(d.o() == 2 and d.e() == 5 and d.n() == 7 for d in [d1, d2, d3, d4, d5])


def test_d_factory_errors():
    """Test D factory function error cases."""
    # Test insufficient parameters
    with pytest.raises(ValueError, match="Must specify exactly two"):
        D(o=2)  # Only one parameter

    with pytest.raises(ValueError, match="Must specify exactly two"):
        D()  # No parameters

    # Test negative results
    with pytest.raises(ValueError, match="results in e=-2 < 0"):
        D(o=5, n=3)  # e = n - o = 3 - 5 = -2

    with pytest.raises(ValueError, match="results in o=-2 < 0"):
        D(e=5, n=3)  # o = n - e = 3 - 5 = -2

    # Test inconsistent parameters
    with pytest.raises(ValueError, match="Inconsistent parameters"):
        D(o=2, e=5, n=6)  # n != o + e (6 != 2 + 5)


def test_d_mathematical_operations():
    """Test mathematical operations."""
    d_obj = D(2, 5)

    # Test new c() method
    c_symbolic = d_obj.c()
    assert isinstance(c_symbolic, sy.Expr)
    print(f"D(2, 5).c(): {c_symbolic}")

    # Test c() with basis
    c_numeric = d_obj.encode(B.Collatz).c()
    assert isinstance(c_numeric, (int, float))
    print(f"D(2, 5) Collatz c(): {c_numeric}")

    # Test new r() method
    r_symbolic = d_obj.r()
    assert isinstance(r_symbolic, sy.Expr)
    print(f"D(2, 5).r(): {r_symbolic}")

    # Test r() with basis
    r_numeric = d_obj.encode(B.Collatz).r()
    assert isinstance(r_numeric, (int, float))
    print(f"D(2, 5) Collatz r(): {r_numeric}")

    # Test new G() method
    g_matrix = d_obj.G()
    assert isinstance(g_matrix, sy.Matrix)
    print(f"D(2, 5).G(): {g_matrix}")
    
    # Verify G() matrix structure for o=2
    assert g_matrix.shape == (2, 1)  # Column vector with 2 elements
    assert g_matrix[0, 0] == sy.Symbol('g')  # g^1
    assert g_matrix[1, 0] == 1  # g^0 = 1
    
    # Test G() with different o values
    d_obj_3 = D(3, 4)  # o=3
    g_matrix_3 = d_obj_3.G()
    assert g_matrix_3.shape == (3, 1)
    assert g_matrix_3[0, 0] == sy.Symbol('g')**2  # g^2
    assert g_matrix_3[1, 0] == sy.Symbol('g')     # g^1
    assert g_matrix_3[2, 0] == 1                  # g^0
    
    # Test G() with o=0 (edge case)
    d_obj_0 = D(0, 3)  # o=0
    g_matrix_0 = d_obj_0.G()
    assert g_matrix_0.shape == (0, 0)  # Empty matrix
    assert len(g_matrix_0) == 0


def test_d_edge_cases():
    """Test edge cases for D class."""
    edge_cases = [(1, 0), (1, 1), (0, 1), (0, 2)]  # (o, e) pairs

    for o_val, e_val in edge_cases:
        print(f"\n=== Testing D({o_val}, {e_val}) ===")
        d_obj = D(o_val, e_val)

        # Basic properties
        assert d_obj.n() >= 0
        assert d_obj.o() >= 0
        assert d_obj.e() >= 0
        assert d_obj.n() == d_obj.o() + d_obj.e()

        # Polynomial evaluation
        d_symbolic = d_obj.d()
        d_numeric = d_obj.encode(B.Collatz).d()

        assert isinstance(d_symbolic, sy.Expr)
        assert isinstance(d_numeric, (int, float, sy.Rational, sy.Integer, sy.Float))

        print(f"D({o_val}, {e_val}) = {d_symbolic}")
        print(f"D({o_val}, {e_val}) with Collatz basis = {d_numeric}")


def test_d_new_methods():
    """Test new c() and r() methods."""
    d_obj = D(2, 5)

    # Test c() method (ceiling)
    c_val = d_obj.c()
    assert isinstance(c_val, sy.Expr)
    print(f"D(2, 5).c(): {c_val}")

    # Test c() with Collatz basis
    c_collatz = d_obj.encode(B.Collatz).c()
    assert isinstance(c_collatz, (int, float))
    print(f"D(2, 5) Collatz c(): {c_collatz}")

    # Test r() method (remainder)
    r_val = d_obj.r()
    assert isinstance(r_val, sy.Expr)
    print(f"D(2, 5).r(): {r_val}")

    # Test r() with Collatz basis
    r_collatz = d_obj.encode(B.Collatz).r()
    assert isinstance(r_collatz, (int, float))
    print(f"D(2, 5) Collatz r(): {r_collatz}")

    # Test relationship: r = c * o - e
    assert r_collatz == c_collatz * d_obj.o() - d_obj.e()


if __name__ == "__main__":
    # Run tests individually for debugging
    test_d_class_creation()
    test_d_from_counts_factory()
    test_d_caching()
    test_d_polynomial_evaluation()
    test_d_as_expr()
    test_d_str_repr()
    test_d_equality()
    test_d_mathematical_operations()
    test_d_edge_cases()
    test_d_quotient_remainder()
    print("\nSimplified D class tests completed!")
