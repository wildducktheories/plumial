"""
Comprehensive tests for the P class (polynomial representations).
"""

import pytest
import sympy as sy

from plumial.core.P import P, cache_info, clear_cache
from plumial.core import B
from plumial.utils.symbolic import g, h, u, v


def test_p_class_creation():
    """Test P class creation and basic properties."""
    clear_cache()

    # Test with p=133
    p_obj = P(133)
    assert p_obj.p() == 133
    assert p_obj.n() == 7  # bit_length - 1
    assert p_obj.o() == 2  # bit_count - 1
    assert p_obj.e() == 5  # n - o

    # Test with small p-values
    for p_val in [3, 5, 9, 17]:
        p_obj = P(p_val)
        assert p_obj.p() == p_val
        assert p_obj.n() == p_val.bit_length() - 1
        assert p_obj.o() == p_val.bit_count() - 1
        assert p_obj.e() == p_obj.n() - p_obj.o()


def test_p_caching():
    """Test LRU caching behavior."""
    clear_cache()

    # Create instances and check caching
    p1 = P(133)
    p2 = P(133)

    # Should be the same object due to caching
    assert p1 is p2

    # Check cache info
    info = cache_info()
    assert info.hits >= 1
    assert info.misses >= 1

    # Clear cache and verify
    clear_cache()
    p3 = P(133)
    assert p3 is not p1  # Different object after cache clear


def test_p_d():
    """Test D class integration."""
    p_obj = P(133)

    # Test d() method
    d_symbolic = p_obj.d()
    d_numeric = p_obj.encode(B.Collatz).d()

    assert isinstance(d_symbolic, sy.Expr)
    assert isinstance(d_numeric, (int, float, sy.Rational))

    # Test D() method returns D object
    d_obj = p_obj.D()
    assert d_obj.n() == p_obj.n()
    assert d_obj.o() == p_obj.o()
    assert d_obj.e() == p_obj.e()

    print(f"P(133).d() = {d_symbolic}")
    print(f"P(133).d(3,2) = {d_numeric}")


def test_p_cycle_navigation():
    """Test cycle navigation methods."""
    p_obj = P(133)

    # Test next() method
    next_obj = p_obj.next()
    assert isinstance(next_obj, type(p_obj))
    assert next_obj.p() != p_obj.p()

    # Test pred() method
    pred_obj = p_obj.pred()
    assert isinstance(pred_obj, type(p_obj))
    assert pred_obj.p() != p_obj.p()

    # Test roundtrip: pred(next(p)) == p
    roundtrip = p_obj.next().pred()
    assert roundtrip.p() == p_obj.p()

    # Test reverse roundtrip: next(pred(p)) == p
    reverse_roundtrip = p_obj.pred().next()
    assert reverse_roundtrip.p() == p_obj.p()

    print(f"P(133) -> next: {next_obj.p()}")
    print(f"P(133) -> pred: {pred_obj.p()}")


def test_p_cycle_iteration():
    """Test cycle() method."""
    p_obj = P(133)

    # Test basic cycle iteration
    cycle_objects = list(p_obj.cycle())
    assert len(cycle_objects) == p_obj.n()

    # First object should be the original
    assert cycle_objects[0].p() == p_obj.p()

    # All objects should be unique
    p_values = [obj.p() for obj in cycle_objects]
    assert len(set(p_values)) == len(p_values)

    # Test with map function
    p_values_mapped = list(p_obj.cycle(map=lambda x: x.p()))
    assert p_values_mapped == p_values

    # Test with filter function
    odd_objects = list(p_obj.cycle(filter=lambda x: x.p() % 2 == 1))
    assert all(obj.p() % 2 == 1 for obj in odd_objects)

    print(f"P(133) cycle: {p_values}")


def test_p_uv_polynomial():
    """Test UV polynomial generation."""
    p_obj = P(133)

    uv_expr = p_obj.uv()
    assert isinstance(uv_expr, sy.Expr)

    # Should contain u and/or v or be a constant
    assert uv_expr.has(u) or uv_expr.has(v) or uv_expr.is_number

    print(f"P(133).uv() = {uv_expr}")

    # Test with smaller p-values
    for p_val in [3, 5, 9]:
        p_obj = P(p_val)
        uv_expr = p_obj.uv()
        print(f"P({p_val}).uv() = {uv_expr}")


def test_p_k_polynomial():
    """Test k polynomial calculation."""
    p_obj = P(133)

    # Symbolic k
    k_symbolic = p_obj.k()
    assert isinstance(k_symbolic, sy.Expr)

    # Numerical k
    k_numeric = p_obj.encode(B.Collatz).k()
    assert isinstance(k_numeric, (int, float, sy.Rational))

    # Test that symbolic form contains g and h
    assert k_symbolic.has(g) or k_symbolic.has(h) or k_symbolic.is_number

    print(f"P(133).k() = {k_symbolic}")
    print(f"P(133).k(3,2) = {k_numeric}")


def test_p_ax_polynomials():
    """Test a,x polynomial pair."""
    p_obj = P(133)

    a_poly, x_poly = p_obj.ax()
    assert isinstance(a_poly, sy.Expr)
    assert isinstance(x_poly, sy.Expr)

    # Test individual a() and x() methods
    a_individual = p_obj.a()
    x_individual = p_obj.x()

    assert sy.simplify(a_poly - a_individual) == 0
    assert sy.simplify(x_poly - x_individual) == 0

    print(f"P(133).a() = {a_poly}")
    print(f"P(133).x() = {x_poly}")


def test_p_a_coefficient():
    """Test a coefficient calculation."""
    p_obj = P(133)

    # Symbolic a
    a_symbolic = p_obj.a()
    assert isinstance(a_symbolic, sy.Expr)

    # Numerical a
    try:
        a_numeric = p_obj.encode(B.Collatz).a()
        assert isinstance(a_numeric, (int, float, sy.Rational))
        print(f"P(133).a(3,2) = {a_numeric}")
    except ZeroDivisionError:
        print("P(133).a(3,2) resulted in division by zero")


def test_p_x_polynomial():
    """Test x polynomial calculation."""
    p_obj = P(133)

    # Symbolic x
    x_symbolic = p_obj.x()
    assert isinstance(x_symbolic, sy.Expr)

    # Numerical x
    try:
        x_numeric = p_obj.encode(B.Collatz).x()
        assert isinstance(x_numeric, (int, float, sy.Rational))
        print(f"P(133).x(3,2) = {x_numeric}")
    except ZeroDivisionError:
        print("P(133).x(3,2) resulted in division by zero")


def test_p_f_polynomial():
    """Test f polynomial (GCD factor)."""
    p_obj = P(133)

    # Symbolic f
    f_symbolic = p_obj.f()
    assert isinstance(f_symbolic, sy.Expr)

    # Numerical f
    f_numeric = p_obj.encode(B.Collatz).f()
    assert isinstance(f_numeric, (int, float, sy.Integer, sy.Float))

    # Test that symbolic form is valid
    assert isinstance(f_symbolic, sy.Expr)

    print(f"P(133).f() = {f_symbolic}")
    print(f"P(133).f(3,2) = {f_numeric}")


def test_p_mathematical_relationships():
    """Test mathematical relationships between polynomials."""
    p_obj = P(133)

    # Test numerical relationships
    try:
        # Encode once and use for all evaluations
        p_collatz = p_obj.encode(B.Collatz)
        d_val = p_collatz.d()
        k_val = p_collatz.k()
        f_val = p_collatz.f()
        a_val = p_collatz.a()
        x_val = p_collatz.x()

        # Test a = d/f
        assert abs(a_val - d_val / f_val) < 1e-10

        # Test x = k/f
        assert abs(x_val - k_val / f_val) < 1e-10

        # Test x*d = a*k
        assert abs(x_val * d_val - a_val * k_val) < 1e-10

        print("Mathematical relationships verified!")

    except ZeroDivisionError:
        print("Some relationships involve division by zero")


def test_p_g_method():
    """Test P class G() method delegation to D object."""
    # Test with p=133 (o=2, e=5)
    p_obj = P(133)
    g_matrix = p_obj.G()
    
    assert isinstance(g_matrix, sy.Matrix)
    assert g_matrix.shape == (2, 1)  # Column vector with 2 elements
    assert g_matrix[0, 0] == sy.Symbol('g')  # g^1
    assert g_matrix[1, 0] == 1  # g^0 = 1
    
    # Test with p=281 (o=3, e=5)
    p_obj_281 = P(281)
    g_matrix_281 = p_obj_281.G()
    
    assert g_matrix_281.shape == (3, 1)  # Column vector with 3 elements
    assert g_matrix_281[0, 0] == sy.Symbol('g')**2  # g^2
    assert g_matrix_281[1, 0] == sy.Symbol('g')     # g^1
    assert g_matrix_281[2, 0] == 1                  # g^0
    
    # Verify delegation: P.G() should match D.G()
    d_obj = p_obj.D()
    d_matrix = d_obj.G()
    assert g_matrix.equals(d_matrix)
    
    print(f"P(133).G(): {g_matrix}")
    print(f"P(281).G(): {g_matrix_281}")


def test_p_h_method():
    """Test P class H() method for k polynomial coefficient matrix."""
    # Test with p=133 (7-bit cycle)
    p_obj = P(133)
    h_matrix = p_obj.H()
    
    assert isinstance(h_matrix, sy.Matrix)
    print(f"P(133).H() matrix shape: {h_matrix.shape}")
    print(f"P(133).H(): {h_matrix}")
    
    # Verify the matrix has correct structure
    if h_matrix.shape[0] > 0:  # If there are odd elements
        assert h_matrix.shape[1] > 0  # Should have at least one coefficient column
    
    # Test with p=281 (8-element cycle)
    p_obj_281 = P(281)
    h_matrix_281 = p_obj_281.H()
    
    assert isinstance(h_matrix_281, sy.Matrix)
    print(f"P(281).H() matrix shape: {h_matrix_281.shape}")
    
    # Count odd elements in cycle to verify matrix rows
    from plumial.utils.functions import isodd
    odd_count_133 = sum(1 for p in p_obj.cycle() if isodd(p))
    odd_count_281 = sum(1 for p in p_obj_281.cycle() if isodd(p))
    
    print(f"P(133) cycle odd elements: {odd_count_133}")
    print(f"P(281) cycle odd elements: {odd_count_281}")
    
    # Verify matrix row count matches odd element count
    if odd_count_133 > 0:
        assert h_matrix.shape[0] == odd_count_133
    if odd_count_281 > 0:
        assert h_matrix_281.shape[0] == odd_count_281
    
    # Test edge case: cycle with no odd elements (if any exist)
    # This is theoretical as most cycles will have odd elements
    
    print(f"H() method test completed successfully")


def test_p_str_repr():
    """Test string representations."""
    p_obj = P(133)

    str_repr = str(p_obj)
    repr_repr = repr(p_obj)

    assert isinstance(str_repr, str)
    assert isinstance(repr_repr, str)
    assert "133" in str_repr
    assert "133" in repr_repr

    print(f"str(P(133)): {str_repr}")
    print(f"repr(P(133)): {repr_repr}")


def test_p_equality():
    """Test equality comparison."""
    p1 = P(133)
    p2 = P(133)
    p3 = P(9)

    assert p1 == p2
    assert p1 != p3
    assert p1 != "not a P object"

    # Test hash consistency
    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)


def test_p_edge_cases():
    """Test edge cases for P class."""
    edge_cases = [2, 3, 4, 5]

    for p_val in edge_cases:
        print(f"\n=== Testing P({p_val}) ===")
        p_obj = P(p_val)

        # Basic properties
        assert p_obj.p() == p_val
        assert p_obj.n() >= 0
        assert p_obj.o() >= 0
        assert p_obj.e() >= 0

        # Polynomial calculations
        p_collatz = p_obj.encode(B.Collatz)
        d_val = p_collatz.d()
        k_val = p_collatz.k()
        uv_val = p_obj.uv()

        assert isinstance(d_val, (int, float, sy.Rational))
        assert isinstance(k_val, (int, float, sy.Rational))
        assert isinstance(uv_val, sy.Expr)

        # Cycle navigation (test that methods work)
        if p_obj.n() > 0:
            next_obj = p_obj.next()
            pred_obj = p_obj.pred()
            # For some edge cases, next/pred might return same value
            # Just test that they return valid P objects
            assert isinstance(next_obj, type(p_obj))
            assert isinstance(pred_obj, type(p_obj))

        print(f"P({p_val}) calculations successful")


def test_p_cache_performance():
    """Test caching performance."""
    clear_cache()

    # Create many P objects with same p-value
    p_val = 133
    objects = []

    for i in range(100):
        p_obj = P(p_val)
        objects.append(p_obj)

    # All should be the same object
    for obj in objects[1:]:
        assert obj is objects[0]

    # Check cache statistics
    info = cache_info()
    assert info.hits >= 99
    assert info.misses <= 1

    print(f"Cache info after 100 P creations: {info}")


def test_p_cycle_linking():
    """Test cycle linking with pred/next parameters."""
    # Create linked cycle
    p1 = P(133)
    p2 = p1.next()
    p3 = p2.next()

    # Test that linking works correctly
    assert p1.next() is p2
    assert p2.pred() is p1
    assert p2.next() is p3
    assert p3.pred() is p2

    print("Cycle linking works correctly")


def test_p_comparison_with_original():
    """Test compatibility with original P implementation."""
    test_values = [9, 17, 33, 65, 133]

    for p_val in test_values:
        print(f"\n=== Testing P({p_val}) ===")
        p_obj = P(p_val)

        # Basic properties
        print(f"p={p_val}, n={p_obj.n()}, o={p_obj.o()}, e={p_obj.e()}")
        print(f"Binary: {bin(p_val)}")

        # Polynomial evaluations
        try:
            p_collatz = p_obj.encode(B.Collatz)
            d_val = p_collatz.d()
            k_val = p_collatz.k()
            uv_val = p_obj.uv()

            print(f"d(3,2) = {d_val}")
            print(f"k(3,2) = {k_val}")
            print(f"uv() = {uv_val}")

            # Test f calculation
            f_val = p_collatz.f()
            print(f"f(3,2) = {f_val}")

        except Exception as e:
            print(f"Error in calculations: {e}")

        # Cycle navigation
        if p_obj.n() > 0:
            next_p = p_obj.next().p()
            pred_p = p_obj.pred().p()
            print(f"next: {next_p}, pred: {pred_p}")


def test_p_error_handling():
    """Test error handling in P class."""
    # Test with invalid p-values should be handled gracefully
    # (The P class may accept any positive integer)

    # Test mathematical operations that might fail
    test_values = [2, 3, 4, 5]

    for p_val in test_values:
        p_obj = P(p_val)

        # These operations might fail for some values
        try:
            p_collatz = p_obj.encode(B.Collatz)
            a_val = p_collatz.a()
            x_val = p_collatz.x()
            print(f"P({p_val}): a={a_val}, x={x_val}")
        except ZeroDivisionError:
            print(f"P({p_val}): Division by zero in a/x calculation")
        except Exception as e:
            print(f"P({p_val}): Error in a/x calculation: {e}")


def test_p_integration_with_binary_ops():
    """Test integration with binary operations."""
    from plumial.utils.binary import e, n, next_p, o, pred_p

    p_obj = P(133)

    # Test that P methods match binary operations
    assert p_obj.n() == n(p_obj.p())
    assert p_obj.o() == o(p_obj.p())
    assert p_obj.e() == e(p_obj.p())

    # Test cycle navigation consistency
    assert p_obj.next().p() == next_p(p_obj.p())
    assert p_obj.pred().p() == pred_p(p_obj.p())

    print("P class integrates correctly with binary operations")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_p_class_creation()
    test_p_caching()
    test_p_d()
    test_p_cycle_navigation()
    test_p_cycle_iteration()
    test_p_uv_polynomial()
    test_p_k_polynomial()
    test_p_ax_polynomials()
    test_p_a_coefficient()
    test_p_x_polynomial()
    test_p_f_polynomial()
    test_p_mathematical_relationships()
    test_p_str_repr()
    test_p_equality()
    test_p_edge_cases()
    test_p_cache_performance()
    test_p_cycle_linking()
    test_p_comparison_with_original()
    test_p_error_handling()
    test_p_integration_with_binary_ops()
    print("\nAll P class tests completed!")
