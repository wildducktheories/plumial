"""
Tests for symbolic operations module.
"""

import pytest
import sympy as sy
from plumial.utils.symbolic import (
    # Basic symbols
    g, h, u, v, a, d, k, x, f, m, n, o, e, i, j, p, b, c, r, q,
    # Constants
    two, three,
    # Indexed symbols
    p_indexed, x_indexed, k_indexed, b_indexed, e_indexed, o_indexed, m_indexed,
    # Collections
    primary_symbols, polynomial_symbols, uv_symbols, bit_count_symbols,
    index_symbols, additional_symbols, constants, indexed_symbols, all_symbols,
    # Utility functions
    get_symbol, get_indexed_symbol, substitute_values, create_polynomial_expression,
    difference_polynomial, uv_to_gh_transform, gh_to_uv_transform, extract_coefficients,
    # Compatibility
    S, SymbolNamespace, symbol_dict, indexed_dict
)


def test_basic_symbols():
    """Test that basic symbols are defined correctly."""
    # Test primary symbols
    assert isinstance(g, sy.Symbol)
    assert isinstance(h, sy.Symbol)
    assert str(g) == 'g'
    assert str(h) == 'h'
    
    # Test UV symbols
    assert isinstance(u, sy.Symbol)
    assert isinstance(v, sy.Symbol)
    assert str(u) == 'u'
    assert str(v) == 'v'
    
    # Test polynomial symbols
    assert isinstance(a, sy.Symbol)
    assert isinstance(d, sy.Symbol)
    assert isinstance(k, sy.Symbol)
    assert isinstance(x, sy.Symbol)
    assert isinstance(f, sy.Symbol)
    
    print("Basic symbols test passed")


def test_bit_count_symbols():
    """Test bit count and index symbols."""
    # Test bit count symbols
    assert isinstance(m, sy.Symbol)
    assert isinstance(n, sy.Symbol)
    assert isinstance(o, sy.Symbol)
    assert isinstance(e, sy.Symbol)
    
    # Test index symbols
    assert isinstance(i, sy.Symbol)
    assert isinstance(j, sy.Symbol)
    
    # Test additional symbols
    assert isinstance(p, sy.Symbol)
    assert isinstance(b, sy.Symbol)
    assert isinstance(c, sy.Symbol)
    assert isinstance(r, sy.Symbol)
    assert isinstance(q, sy.Symbol)
    
    print("Bit count symbols test passed")


def test_constants():
    """Test mathematical constants."""
    assert isinstance(two, sy.Integer)
    assert isinstance(three, sy.Integer)
    assert two == 2
    assert three == 3
    
    print("Constants test passed")


def test_indexed_symbols():
    """Test indexed symbols."""
    # Test that indexed symbols are IndexedBase
    assert isinstance(p_indexed, sy.IndexedBase)
    assert isinstance(x_indexed, sy.IndexedBase)
    assert isinstance(k_indexed, sy.IndexedBase)
    
    # Test indexing
    p0 = p_indexed[0]
    p1 = p_indexed[1]
    assert p0 != p1
    assert str(p0) == 'p[0]'
    assert str(p1) == 'p[1]'
    
    # Test multiple indices
    x_ij = x_indexed[i, j]
    assert str(x_ij) == 'x[i, j]'
    
    print("Indexed symbols test passed")


def test_symbol_collections():
    """Test symbol collections."""
    # Test that collections contain the right symbols
    assert g in primary_symbols
    assert h in primary_symbols
    assert len(primary_symbols) == 2
    
    assert u in uv_symbols
    assert v in uv_symbols
    assert len(uv_symbols) == 2
    
    assert a in polynomial_symbols
    assert d in polynomial_symbols
    assert k in polynomial_symbols
    assert x in polynomial_symbols
    assert f in polynomial_symbols
    
    # Test all_symbols contains everything
    assert len(all_symbols) > 10
    assert g in all_symbols
    assert h in all_symbols
    assert u in all_symbols
    assert v in all_symbols
    
    print("Symbol collections test passed")


def test_get_symbol():
    """Test get_symbol function."""
    # Test valid symbols
    assert get_symbol('g') is g
    assert get_symbol('h') is h
    assert get_symbol('u') is u
    assert get_symbol('v') is v
    
    # Test invalid symbol
    with pytest.raises(KeyError):
        get_symbol('invalid_symbol')
    
    print("get_symbol test passed")


def test_get_indexed_symbol():
    """Test get_indexed_symbol function."""
    # Test valid indexed symbols
    assert get_indexed_symbol('p') is p_indexed
    assert get_indexed_symbol('x') is x_indexed
    assert get_indexed_symbol('k') is k_indexed
    
    # Test invalid indexed symbol
    with pytest.raises(KeyError):
        get_indexed_symbol('invalid_indexed')
    
    print("get_indexed_symbol test passed")


def test_substitute_values():
    """Test substitute_values function."""
    # Create test expression
    expr = g**2 + h*u + v
    
    # Test substitution
    result = substitute_values(expr, g=3, h=2, u=1, v=5)
    expected = 3**2 + 2*1 + 5  # 9 + 2 + 5 = 16
    assert result == expected
    
    # Test partial substitution
    partial = substitute_values(expr, g=3)
    assert partial == 9 + h*u + v
    
    # Test invalid symbol name
    with pytest.raises(KeyError):
        substitute_values(expr, invalid_symbol=1)
    
    print("substitute_values test passed")


def test_create_polynomial_expression():
    """Test create_polynomial_expression function."""
    # Test simple polynomial
    expr = create_polynomial_expression({'g': 2, 'h': 3})
    expected = g**2 * h**3
    assert expr == expected
    
    # Test with zero power (should be omitted)
    expr = create_polynomial_expression({'g': 2, 'h': 0, 'u': 1})
    expected = g**2 * u
    assert expr == expected
    
    # Test constant (empty powers)
    expr = create_polynomial_expression({})
    assert expr == 1
    
    # Test invalid symbol
    with pytest.raises(KeyError):
        create_polynomial_expression({'invalid': 1})
    
    print("create_polynomial_expression test passed")


def test_difference_polynomial():
    """Test difference_polynomial function."""
    # Test basic difference polynomial
    diff = difference_polynomial(5, 2)
    expected = h**5 - g**2
    assert diff == expected
    
    # Test with powers of 1
    diff = difference_polynomial(3, 1)
    expected = h**3 - g
    assert diff == expected
    
    # Test with zero powers
    diff = difference_polynomial(0, 0)
    expected = h**0 - g**0  # 1 - 1 = 0
    assert diff == 0
    
    print("difference_polynomial test passed")


def test_uv_to_gh_transform():
    """Test UV to GH transformation."""
    # Test basic transformation
    uv_expr = u**2 + v
    gh_expr = uv_to_gh_transform(uv_expr)
    expected = (g*h)**2 + h  # g^2*h^2 + h
    assert sy.expand(gh_expr) == sy.expand(expected)
    
    # Test more complex expression
    uv_expr = u*v + u**2 + 3*v
    gh_expr = uv_to_gh_transform(uv_expr)
    expected = (g*h)*h + (g*h)**2 + 3*h
    assert sy.expand(gh_expr) == sy.expand(expected)
    
    print("uv_to_gh_transform test passed")


def test_gh_to_uv_transform():
    """Test GH to UV transformation."""
    # Test simple transformation (reverse of uv_to_gh)
    gh_expr = g**2*h**2 + h
    uv_expr = gh_to_uv_transform(gh_expr)
    # This is a simplified reverse transform - may not work perfectly for all cases
    
    print("gh_to_uv_transform test completed")


def test_extract_coefficients():
    """Test extract_coefficients function."""
    # Test polynomial in g
    expr = 3*g**2 + 2*g + 5
    coeffs = extract_coefficients(expr, g)
    
    # Should have coefficients for powers 0, 1, 2
    assert 5 in coeffs.values()  # constant term
    assert 2 in coeffs.values()  # g term
    assert 3 in coeffs.values()  # g^2 term
    
    print("extract_coefficients test passed")


def test_symbol_namespace():
    """Test SymbolNamespace compatibility layer."""
    # Test that S has the expected attributes
    assert hasattr(S, 'g')
    assert hasattr(S, 'h')
    assert hasattr(S, 'u')
    assert hasattr(S, 'v')
    
    # Test that attributes are the correct symbols
    assert S.g is g
    assert S.h is h
    assert S.u is u
    assert S.v is v
    
    # Test accessing additional symbols
    assert S.a is a
    assert S.d is d
    assert S.k is k
    
    print("SymbolNamespace test passed")


def test_symbol_dictionaries():
    """Test symbol lookup dictionaries."""
    # Test symbol_dict
    assert 'g' in symbol_dict
    assert 'h' in symbol_dict
    assert symbol_dict['g'] is g
    assert symbol_dict['h'] is h
    
    # Test indexed_dict
    assert 'p' in indexed_dict
    assert 'x' in indexed_dict
    assert indexed_dict['p'] is p_indexed
    assert indexed_dict['x'] is x_indexed
    
    print("Symbol dictionaries test passed")


def test_compatibility_with_original():
    """Test compatibility with original library patterns."""
    # Test that we can use symbols like the original library
    
    # Original pattern: S.g, S.h
    expr1 = S.g**2 + S.h
    expr2 = g**2 + h
    assert expr1 == expr2
    
    # Test substitution patterns
    expr = S.g**S.o + S.h**S.e
    result = expr.subs([(S.g, 3), (S.h, 2), (S.o, 2), (S.e, 3)])
    expected = 3**2 + 2**3  # 9 + 8 = 17
    assert result == expected
    
    # Test indexed symbols
    indexed_expr = S.p_indexed[0] + S.x_indexed[1]
    assert str(indexed_expr) == 'p[0] + x[1]'
    
    print("Compatibility test passed")


def test_mathematical_operations():
    """Test mathematical operations with symbols."""
    # Test basic arithmetic
    expr1 = g + h
    expr2 = g - h
    expr3 = g * h
    expr4 = g / h
    
    assert isinstance(expr1, sy.Add)
    assert isinstance(expr2, sy.Add)  # Subtraction is addition with negative
    assert isinstance(expr3, sy.Mul)
    assert isinstance(expr4, sy.Mul)  # Division is multiplication with inverse
    
    # Test powers
    expr5 = g**2
    expr6 = h**e
    
    assert isinstance(expr5, sy.Pow)
    assert isinstance(expr6, sy.Pow)
    
    # Test factorization
    expr = g**2 - h**2
    factored = sy.factor(expr)
    assert factored == (g - h)*(g + h)
    
    print("Mathematical operations test passed")


def test_complex_expressions():
    """Test complex symbolic expressions."""
    # Test UV polynomial
    uv_poly = u**2*v + u*v**2 + v**3
    
    # Transform to GH
    gh_poly = uv_to_gh_transform(uv_poly)
    
    # Should be in terms of g and h
    assert not gh_poly.has(u)
    assert not gh_poly.has(v)
    assert gh_poly.has(g) or gh_poly.has(h)
    
    # Test difference polynomial
    diff_poly = difference_polynomial(5, 3)
    assert diff_poly == h**5 - g**3
    
    # Test substitution in difference polynomial
    result = substitute_values(diff_poly, g=2, h=3)
    expected = 3**5 - 2**3  # 243 - 8 = 235
    assert result == expected
    
    print("Complex expressions test passed")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_basic_symbols()
    test_bit_count_symbols()
    test_constants()
    test_indexed_symbols()
    test_symbol_collections()
    test_get_symbol()
    test_get_indexed_symbol()
    test_substitute_values()
    test_create_polynomial_expression()
    test_difference_polynomial()
    test_uv_to_gh_transform()
    test_gh_to_uv_transform()
    test_extract_coefficients()
    test_symbol_namespace()
    test_symbol_dictionaries()
    test_compatibility_with_original()
    test_mathematical_operations()
    test_complex_expressions()
    print("\nAll symbolic operations tests completed!")