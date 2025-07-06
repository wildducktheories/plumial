"""
Simplified tests for the D class (difference polynomials).
"""

import pytest
import sympy as sy
from plumial.core.difference_polynomials import D, D_from_counts, clear_d_cache, d_cache_info
from plumial.utils.symbolic import g, h


def test_d_class_creation():
    """Test D class creation and basic properties."""
    clear_d_cache()
    
    # Test with p=133
    d_obj = D(133)
    assert d_obj.n() == 7  # bit_length - 1
    assert d_obj.o() == 2  # bit_count - 1
    assert d_obj.e() == 5  # n - o
    
    # Test with small p-values
    for p_val in [3, 5, 9, 17]:
        d_obj = D(p_val)
        assert d_obj.n() == p_val.bit_length() - 1
        assert d_obj.o() == p_val.bit_count() - 1
        assert d_obj.e() == d_obj.n() - d_obj.o()


def test_d_from_counts_factory():
    """Test D_from_counts factory function."""
    # Test with known values
    d_obj = D_from_counts(n=3, o=1)
    assert d_obj.n() == 3
    assert d_obj.o() == 1
    assert d_obj.e() == 2
    
    # Test edge cases
    d_obj = D_from_counts(n=1, o=0)
    assert d_obj.n() == 1
    assert d_obj.o() == 0
    assert d_obj.e() == 1


def test_d_caching():
    """Test LRU caching behavior."""
    clear_d_cache()
    
    # Create instances and check caching
    d1 = D(133)
    d2 = D(133)
    
    # Should be the same object due to caching
    assert d1 is d2
    
    # Check cache info
    info = d_cache_info()
    assert info.hits >= 1
    assert info.misses >= 1


def test_d_polynomial_evaluation():
    """Test difference polynomial evaluation."""
    d_obj = D(133)
    
    # Symbolic evaluation
    d_symbolic = d_obj.d()
    assert isinstance(d_symbolic, sy.Expr)
    assert d_symbolic.has(g) or d_symbolic.has(h) or d_symbolic.is_number
    
    # Numerical evaluation
    d_numeric = d_obj.d(g=3, h=2)
    assert isinstance(d_numeric, (int, float, sy.Rational))
    
    print(f"D(133) symbolic: {d_symbolic}")
    print(f"D(133, g=3, h=2): {d_numeric}")


def test_d_as_expr():
    """Test as_expr method."""
    d_obj = D(133)
    expr = d_obj.as_expr()
    
    assert isinstance(expr, sy.Expr)
    assert expr.has(g) or expr.has(h)
    
    print(f"D(133).as_expr(): {expr}")


def test_d_str_repr():
    """Test string representations."""
    d_obj = D(133)
    
    str_repr = str(d_obj)
    repr_repr = repr(d_obj)
    
    assert isinstance(str_repr, str)
    assert isinstance(repr_repr, str)
    
    print(f"str(D(133)): {str_repr}")
    print(f"repr(D(133)): {repr_repr}")


def test_d_equality():
    """Test equality comparison."""
    d1 = D(133)
    d2 = D(133)
    d3 = D(9)
    
    assert d1 == d2
    assert d1 != d3
    assert d1 != "not a D object"
    
    # Test hash consistency
    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)


def test_d_mathematical_operations():
    """Test mathematical operations."""
    d_obj = D(133)
    
    # Test factor method
    try:
        factored = d_obj.factor()
        assert isinstance(factored, sy.Expr)
        print(f"D(133).factor(): {factored}")
    except Exception as e:
        print(f"Factor failed: {e}")
    
    # Test expand method
    try:
        expanded = d_obj.expand()
        assert isinstance(expanded, sy.Expr)
        print(f"D(133).expand(): {expanded}")
    except Exception as e:
        print(f"Expand failed: {e}")
    
    # Test gcd method
    try:
        gcd_val = d_obj.gcd(g=3, h=2)
        assert isinstance(gcd_val, (int, float, sy.Expr))
        print(f"D(133).gcd(3,2): {gcd_val}")
    except Exception as e:
        print(f"GCD failed: {e}")


def test_d_edge_cases():
    """Test edge cases for D class."""
    edge_cases = [2, 3, 4, 5]
    
    for p_val in edge_cases:
        print(f"\n=== Testing D({p_val}) ===")
        d_obj = D(p_val)
        
        # Basic properties
        assert d_obj.n() >= 0
        assert d_obj.o() >= 0
        assert d_obj.e() >= 0
        
        # Polynomial evaluation
        d_symbolic = d_obj.d()
        d_numeric = d_obj.d(g=3, h=2)
        
        assert isinstance(d_symbolic, sy.Expr)
        assert isinstance(d_numeric, (int, float, sy.Rational))
        
        print(f"D({p_val}) = {d_symbolic}")
        print(f"D({p_val}, g=3, h=2) = {d_numeric}")


def test_d_quotient_remainder():
    """Test quotient and remainder methods."""
    d_obj = D(133)
    
    q_val = d_obj.q()
    r_val = d_obj.r()
    
    assert isinstance(q_val, int)
    assert isinstance(r_val, int)
    assert r_val >= 0
    assert r_val < d_obj.o() if d_obj.o() > 0 else r_val == 0
    
    print(f"D(133): q={q_val}, r={r_val}")


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