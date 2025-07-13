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
    """Test private _as_expr method."""
    d_obj = D(2, 5)
    expr = d_obj._as_expr()

    assert isinstance(expr, sy.Expr)
    assert expr.has(g) or expr.has(h)

    print(f"D(2, 5)._as_expr(): {expr}")


def test_d_sympy_integration():
    """Test SymPy integration methods for _D class."""
    d1 = D(2, 5)  # h^5 - g^2
    d2 = D(1, 3)  # h^3 - g^1
    x = sy.Symbol('x')
    
    # Test _sympy_() conversion
    expr1 = d1._sympy_()
    assert isinstance(expr1, sy.Expr)
    assert expr1 == h**5 - g**2
    
    # Test addition with D objects
    sum_dd = d1 + d2
    assert isinstance(sum_dd, sy.Expr)
    assert sum_dd == (h**5 - g**2) + (h**3 - g)
    assert sy.expand(sum_dd) == h**5 + h**3 - g**2 - g
    
    # Test addition with SymPy expressions
    sum_dx = d1 + x
    assert isinstance(sum_dx, sy.Expr)
    assert sum_dx == (h**5 - g**2) + x
    
    # Test right addition (x + D)
    sum_xd = x + d1
    assert isinstance(sum_xd, sy.Expr)
    assert sum_xd == x + (h**5 - g**2)
    assert sum_xd == sum_dx  # Should be commutative
    
    # Test subtraction with D objects
    diff_dd = d1 - d2
    assert isinstance(diff_dd, sy.Expr)
    assert diff_dd == (h**5 - g**2) - (h**3 - g)
    assert sy.expand(diff_dd) == h**5 - h**3 - g**2 + g
    
    # Test subtraction with SymPy expressions
    diff_dx = d1 - x
    assert isinstance(diff_dx, sy.Expr)
    assert diff_dx == (h**5 - g**2) - x
    
    # Test right subtraction (x - D)
    diff_xd = x - d1
    assert isinstance(diff_xd, sy.Expr)
    assert diff_xd == x - (h**5 - g**2)
    
    # Test multiplication with D objects
    prod_dd = d1 * d2
    assert isinstance(prod_dd, sy.Expr)
    assert prod_dd == (h**5 - g**2) * (h**3 - g)
    
    # Test multiplication with SymPy expressions
    prod_dx = d1 * x
    assert isinstance(prod_dx, sy.Expr)
    assert prod_dx == (h**5 - g**2) * x
    
    # Test right multiplication (x * D)
    prod_xd = x * d1
    assert isinstance(prod_xd, sy.Expr)
    assert prod_xd == x * (h**5 - g**2)
    assert prod_xd == prod_dx  # Should be commutative
    
    # Test division with D objects
    div_dd = d1 / d2
    assert isinstance(div_dd, sy.Expr)
    assert div_dd == (h**5 - g**2) / (h**3 - g)
    
    # Test division with SymPy expressions
    div_dx = d1 / x
    assert isinstance(div_dx, sy.Expr)
    assert div_dx == (h**5 - g**2) / x
    
    # Test right division (x / D)
    div_xd = x / d1
    assert isinstance(div_xd, sy.Expr)
    assert div_xd == x / (h**5 - g**2)
    
    # Test exponentiation
    pow_d = d1 ** 2
    assert isinstance(pow_d, sy.Expr)
    assert pow_d == (h**5 - g**2) ** 2
    
    # Test right exponentiation (x ** D)
    pow_xd = x ** d1
    assert isinstance(pow_xd, sy.Expr)
    assert pow_xd == x ** (h**5 - g**2)
    
    # Test negation
    neg_d = -d1
    assert isinstance(neg_d, sy.Expr)
    assert neg_d == -(h**5 - g**2)
    assert neg_d == -h**5 + g**2
    
    # Test positive (unary plus)
    pos_d = +d1
    assert isinstance(pos_d, sy.Expr)
    assert pos_d == +(h**5 - g**2)
    assert pos_d == h**5 - g**2
    
    print(f"D(2, 5) + D(1, 3) = {sum_dd}")
    print(f"D(2, 5) * D(1, 3) = {prod_dd}")
    print(f"D(2, 5) ** 2 = {pow_d}")


def test_d_sympy_complex_expressions():
    """Test D objects in complex SymPy expressions."""
    d1 = D(2, 5)  # h^5 - g^2
    d2 = D(1, 3)  # h^3 - g^1
    x = sy.Symbol('x')
    y = sy.Symbol('y')
    
    # Test complex expressions involving multiple operations
    complex_expr = (d1 + d2) * x - y / d1
    assert isinstance(complex_expr, sy.Expr)
    
    # Test differentiation
    diff_expr = sy.diff(d1 + x**2, x)
    assert isinstance(diff_expr, sy.Expr)
    assert diff_expr == 2*x  # d(h^5 - g^2 + x^2)/dx = 2x
    
    # Test integration (symbolic)
    int_expr = sy.integrate(d1 + x, x)
    assert isinstance(int_expr, sy.Expr)
    expected = sy.integrate(h**5 - g**2 + x, x)
    assert int_expr == expected
    
    # Test substitution
    sub_expr = (d1 + x).subs(x, 3)
    assert isinstance(sub_expr, sy.Expr)
    assert sub_expr == h**5 - g**2 + 3
    
    # Test expansion
    expanded = sy.expand((d1 + d2)**2)
    assert isinstance(expanded, sy.Expr)
    
    # Test simplification
    simplified = sy.simplify(d1 / d1)
    assert simplified == 1
    
    print(f"Complex expression: {complex_expr}")
    print(f"Derivative: {diff_expr}")
    print(f"Integral: {int_expr}")
    print(f"Substitution: {sub_expr}")


def test_d_sympy_basis_integration():
    """Test SymPy integration with different bases."""
    d_symbolic = D(2, 5)  # Symbolic basis
    d_collatz = d_symbolic.encode(B.Collatz)  # Collatz basis (g=3, h=2)
    x = sy.Symbol('x')
    
    # Test that both symbolic and encoded D objects work in expressions
    expr_symbolic = d_symbolic + x
    expr_collatz = d_collatz + x
    
    assert isinstance(expr_symbolic, sy.Expr)
    assert isinstance(expr_collatz, sy.Expr)
    
    # Symbolic should contain g and h symbols
    assert expr_symbolic.has(g) or expr_symbolic.has(h)
    
    # Both should work in mathematical operations
    result_symbolic = sy.expand(expr_symbolic * 2)
    result_collatz = sy.expand(expr_collatz * 2)
    
    assert isinstance(result_symbolic, sy.Expr)
    assert isinstance(result_collatz, sy.Expr)
    
    print(f"Symbolic D + x: {expr_symbolic}")
    print(f"Collatz D + x: {expr_collatz}")
    print(f"2 * (Symbolic D + x): {result_symbolic}")
    print(f"2 * (Collatz D + x): {result_collatz}")


def test_d_sympy_matrix_operations():
    """Test D objects in matrix operations with SymPy."""
    d1 = D(2, 5)  # h^5 - g^2
    d2 = D(1, 3)  # h^3 - g
    
    # Test creating matrices with D objects
    matrix = sy.Matrix([[d1, d2], [d2, d1]])
    assert isinstance(matrix, sy.Matrix)
    assert matrix.shape == (2, 2)
    
    # Verify matrix elements are properly converted
    assert matrix[0, 0] == h**5 - g**2
    assert matrix[0, 1] == h**3 - g
    assert matrix[1, 0] == h**3 - g
    assert matrix[1, 1] == h**5 - g**2
    
    # Test matrix operations
    matrix_squared = matrix**2
    assert isinstance(matrix_squared, sy.Matrix)
    
    # Test matrix multiplication with scalar
    scaled_matrix = 2 * matrix
    assert isinstance(scaled_matrix, sy.Matrix)
    assert scaled_matrix[0, 0] == 2 * (h**5 - g**2)
    
    print(f"Matrix with D objects:\n{matrix}")
    print(f"Scaled matrix:\n{scaled_matrix}")


def test_d_sympy_edge_cases():
    """Test edge cases for SymPy integration."""
    # Test with zero polynomials
    d_zero_o = D(0, 3)  # h^3 - g^0 = h^3 - 1
    d_zero_e = D(2, 0)  # h^0 - g^2 = 1 - g^2
    
    assert isinstance(d_zero_o + 1, sy.Expr)
    assert isinstance(d_zero_e * 2, sy.Expr)
    
    # Test with very small polynomials
    d_minimal = D(1, 1)  # h^1 - g^1 = h - g
    x = sy.Symbol('x')
    
    result = sy.solve(d_minimal + x, x)
    assert len(result) == 1
    assert result[0] == g - h
    
    # Test with identical D objects
    d1 = D(2, 3)
    d2 = D(2, 3)
    
    # Should be able to perform operations
    sum_identical = d1 + d2
    assert sum_identical == 2 * (h**3 - g**2)
    
    # Test subtraction of identical objects
    diff_identical = d1 - d2
    assert diff_identical == 0
    
    print(f"D(0, 3) + 1: {d_zero_o + 1}")
    print(f"D(2, 0) * 2: {d_zero_e * 2}")
    print(f"Solve D(1, 1) + x = 0: {sy.solve(d_minimal + x, x)}")
    print(f"D(2, 3) - D(2, 3): {diff_identical}")


def test_d_repr_consistency_requirement():
    """Test the key requirement: D().__repr__() == (D()*1).__repr__()"""
    from plumial import P
    
    # Test with various P values
    test_p_values = [133, 293, 5, 17, 9]
    
    for p_val in test_p_values:
        d = P(p_val).D()
        
        # This is the key requirement from the user
        assert d.__repr__() == (d * 1).__repr__(), f"Failed for P({p_val}).D()"
        
        # Also test that repr equals the underlying expression repr
        assert repr(d) == repr(d._as_expr()), f"D repr doesn't match expression repr for P({p_val})"
        
        # Test with other operations that should return SymPy expressions
        assert repr(d) == repr(d + 0), f"D + 0 repr mismatch for P({p_val})"
        assert repr(d) == repr(0 + d), f"0 + D repr mismatch for P({p_val})"
    
    # Test with direct D construction
    test_d_params = [(1, 2), (2, 3), (0, 4), (3, 0)]
    
    for o, e in test_d_params:
        d = D(o, e)
        assert d.__repr__() == (d * 1).__repr__(), f"Failed for D({o}, {e})"
        assert repr(d) == repr(d._as_expr()), f"D repr doesn't match expression repr for D({o}, {e})"
    
    print("✓ All D.__repr__() == (D*1).__repr__() requirements satisfied")


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
    test_d_caching()
    test_d_polynomial_evaluation()
    test_d_as_expr()
    test_d_sympy_integration()
    test_d_sympy_complex_expressions()
    test_d_sympy_basis_integration()
    test_d_sympy_matrix_operations()
    test_d_sympy_edge_cases()
    test_d_repr_consistency_requirement()
    test_d_str_repr()
    test_d_equality()
    test_d_factory_patterns()
    test_d_factory_errors()
    test_d_mathematical_operations()
    test_d_edge_cases()
    test_d_new_methods()
    print("\nD class tests with SymPy integration completed!")
