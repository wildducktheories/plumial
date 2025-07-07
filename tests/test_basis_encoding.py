"""
Tests for basis and encoding functionality in P and D classes.

This module tests the new basis-aware architecture including:
- Basis class functionality
- P and D object encoding methods
- Equality semantics with basis consideration
- Round-trip and transitive encoding properties
- Backward compatibility with legacy API
"""

import pytest
import sympy as sy
from plumial.core import P, D, Basis, B


class TestBasisClass:
    """Test the Basis class and its methods."""
    
    def test_basis_creation(self):
        """Test basic basis creation and access methods."""
        basis = Basis(3, 2)
        
        assert basis.g() == 3
        assert basis.h() == 2
        assert basis.odd() == 3
        assert basis.even() == 2
    
    def test_basis_unpacking(self):
        """Test tuple and dict unpacking methods."""
        basis = Basis(5, 4)
        
        assert basis.tuple() == (5, 4)
        assert basis.dict() == {'g': 5, 'h': 4}
    
    def test_basis_equality(self):
        """Test basis equality and hashing."""
        basis1 = Basis(3, 2)
        basis2 = Basis(3, 2)
        basis3 = Basis(5, 2)
        
        assert basis1 == basis2
        assert basis1 != basis3
        assert hash(basis1) == hash(basis2)
        assert hash(basis1) != hash(basis3)
    
    def test_predefined_bases(self):
        """Test predefined basis constants."""
        assert B.Collatz.g() == 3
        assert B.Collatz.h() == 2
        assert B.Collatz_5_2.g() == 5
        assert B.Collatz_5_2.h() == 2
    
    def test_sym_dict_method(self):
        """Test sym_dict method for symbolic substitution."""
        from plumial.utils.symbolic import g, h
        
        # Create a symbolic expression
        expr = g**2 + h**3
        
        # Test with Collatz basis
        result = expr.subs(B.Collatz.sym_dict())
        expected = 3**2 + 2**3  # g=3, h=2
        assert result == expected
        
        # Test with custom basis
        custom_basis = Basis(5, 4)
        result2 = expr.subs(custom_basis.sym_dict())
        expected2 = 5**2 + 4**3  # g=5, h=4
        assert result2 == expected2


class TestPClassEncoding:
    """Test P class basis and encoding functionality."""
    
    def test_default_symbolic_basis(self):
        """Test that P objects default to symbolic basis."""
        p = P(281)
        assert p.basis() == B.Symbolic
    
    def test_basis_construction(self):
        """Test creating P objects with specific basis."""
        p_symbolic = P(281)
        p_collatz = P(281, basis=B.Collatz)
        
        assert p_symbolic.basis() == B.Symbolic
        assert p_collatz.basis() == B.Collatz
        assert p_symbolic != p_collatz  # Different basis means different object
    
    def test_encode_method(self):
        """Test the encode method for basis transformation."""
        p = P(281)
        
        # Encode to different bases
        collatz_p = p.encode(B.Collatz)
        custom_p = p.encode(g=5, h=2)
        
        assert collatz_p.basis() == B.Collatz
        assert custom_p.basis() == Basis(5, 2)
        assert p.p() == collatz_p.p() == custom_p.p()  # Same p-value
    
    def test_round_trip_equality(self):
        """Test round-trip encoding equality."""
        p = P(281)
        collatz_p = p.encode(B.Collatz)
        back_p = collatz_p.encode()  # Empty encode returns to symbolic
        
        assert p != collatz_p  # Different basis
        assert back_p == p     # Round-trip equality
    
    def test_transitive_encoding_equality(self):
        """Test transitive encoding property."""
        a = P(281).encode(g=3, h=2)
        b = a.encode(g=5, h=2)
        c = b.encode(g=3, h=2)
        
        assert a == c  # Transitive property
        assert a.basis() == c.basis()
        assert a.p() == b.p() == c.p()
    
    def test_basis_aware_evaluation(self):
        """Test that encoded objects use their basis automatically."""
        p_symbolic = P(281)
        p_collatz = P(281).encode(B.Collatz)
        
        # Symbolic should remain symbolic
        k_symbolic = p_symbolic.k()
        assert hasattr(k_symbolic, 'subs')  # SymPy expression
        
        # Collatz should evaluate numerically  
        k_collatz = p_collatz.k()
        # SymPy returns sympy.Integer, not Python int
        assert isinstance(k_collatz, (int, float, sy.Integer, sy.Float))  # Numerical result
    
    def test_legacy_compatibility(self):
        """Test backward compatibility with legacy g,h parameters."""
        p = P(281)
        
        # Legacy API should still work
        k_legacy = p.k(g=3, h=2)
        d_legacy = p.d(g=3, h=2)
        
        # Should give same result as encoding
        p_encoded = p.encode(g=3, h=2)
        k_encoded = p_encoded.k()
        d_encoded = p_encoded.d()
        
        assert k_legacy == k_encoded
        assert d_legacy == d_encoded


class TestDClassEncoding:
    """Test D class basis and encoding functionality."""
    
    def test_default_symbolic_basis(self):
        """Test that D objects default to symbolic basis."""
        d = D(281)
        assert d.basis() == B.Symbolic
    
    def test_basis_construction(self):
        """Test creating D objects with specific basis."""
        d_symbolic = D(281)
        d_collatz = D(281, basis=B.Collatz)
        
        assert d_symbolic.basis() == B.Symbolic
        assert d_collatz.basis() == B.Collatz
        assert d_symbolic != d_collatz  # Different basis means different object
    
    def test_encode_method(self):
        """Test the encode method for basis transformation."""
        d = D(281)
        
        # Encode to different bases
        collatz_d = d.encode(B.Collatz)
        custom_d = d.encode(g=5, h=2)
        
        assert collatz_d.basis() == B.Collatz
        assert custom_d.basis() == Basis(5, 2)
        assert d.n() == collatz_d.n() == custom_d.n()  # Same bit counts
        assert d.o() == collatz_d.o() == custom_d.o()
    
    def test_round_trip_equality(self):
        """Test round-trip encoding equality."""
        d = D(281)
        collatz_d = d.encode(B.Collatz)
        back_d = collatz_d.encode()  # Empty encode returns to symbolic
        
        assert d != collatz_d  # Different basis
        assert back_d == d     # Round-trip equality
    
    def test_alternative_construction(self):
        """Test D construction from n,o parameters."""
        d1 = D(281)
        d2 = D(n=8, o=3)  # Same as D(281): p=281 -> n=8, o=3
        d3 = D(n=8, o=3, basis=B.Collatz)
        
        assert d1.n() == d2.n() == 8
        assert d1.o() == d2.o() == 3
        assert d1.basis() == d2.basis() == B.Symbolic
        assert d3.basis() == B.Collatz
    
    def test_basis_aware_evaluation(self):
        """Test that encoded D objects use their basis automatically."""
        d_symbolic = D(281)
        d_collatz = D(281).encode(B.Collatz)
        
        # Symbolic should remain symbolic
        expr_symbolic = d_symbolic.d()
        assert hasattr(expr_symbolic, 'subs')  # SymPy expression
        
        # Collatz should evaluate numerically
        expr_collatz = d_collatz.d()
        # SymPy returns sympy.Integer, not Python int
        assert isinstance(expr_collatz, (int, float, sy.Integer, sy.Float))  # Numerical result


class TestCaching:
    """Test that caching works correctly with basis."""
    
    def test_p_caching_with_basis(self):
        """Test that P objects are cached correctly by (p-value, basis)."""
        p1 = P(133)
        p2 = P(133)
        p3 = P(133, basis=B.Collatz)
        p4 = P(133, basis=B.Collatz)
        
        assert p1 is p2  # Same basis, should be cached
        assert p3 is p4  # Same basis, should be cached
        assert p1 is not p3  # Different basis, different objects
    
    def test_d_caching_with_basis(self):
        """Test that D objects are cached correctly by (n, o, basis)."""
        d1 = D(133)
        d2 = D(133)
        d3 = D(133, basis=B.Collatz)
        d4 = D(133, basis=B.Collatz)
        
        assert d1 is d2  # Same basis, should be cached
        assert d3 is d4  # Same basis, should be cached
        assert d1 is not d3  # Different basis, different objects


class TestBasisPropagation:
    """Test that basis propagates correctly through operations."""
    
    def test_next_preserves_basis(self):
        """Test that next() preserves the basis of the original object."""
        a = P(281).encode(B.Collatz)
        assert a.basis() == a.next().basis()
    
    def test_pred_preserves_basis(self):
        """Test that pred() preserves the basis of the original object."""
        a = P(281).encode(B.Collatz)
        assert a.basis() == a.pred().basis()
    
    def test_cycle_preserves_basis(self):
        """Test that all cycle elements have the same basis."""
        a = P(281).encode(B.Collatz)
        cycle = list(a.cycle())
        
        # All cycle elements should have the same basis as the original
        for element in cycle:
            assert element.basis() == a.basis()


class TestSymbolicIntegration:
    """Test symbolic expression integration with basis."""
    
    def test_gh_polynomial_substitution(self):
        """Test the pattern: gh_poly.subs(B.Collatz.sym_dict())"""
        from plumial.utils.symbolic import g, h
        
        # Create a GH polynomial (like what P.k() might return symbolically)
        gh_poly = g**2 + g*h**2 + h**2
        
        # Use the new pattern to substitute basis values
        result = gh_poly.subs(B.Collatz.sym_dict())
        
        # Verify the result (g=3, h=2)
        expected = 3**2 + 3*(2**2) + 2**2  # 9 + 12 + 4 = 25
        assert result == expected
    
    def test_d_polynomial_substitution(self):
        """Test basis substitution with d-polynomials."""
        from plumial.utils.symbolic import g, h
        
        # Example d-polynomial: h^5 - g^2 (like from P(133))
        d_poly = h**5 - g**2
        
        # Use different bases
        collatz_result = d_poly.subs(B.Collatz.sym_dict())
        five_two_result = d_poly.subs(B.Collatz_5_2.sym_dict())
        
        # Verify results
        assert collatz_result == 2**5 - 3**2  # 32 - 9 = 23
        assert five_two_result == 2**5 - 5**2  # 32 - 25 = 7
    
    def test_mixed_expressions(self):
        """Test with more complex expressions involving both g and h."""
        from plumial.utils.symbolic import g, h
        
        # Complex expression
        expr = g**3 - 2*g*h + h**4 - 7
        
        # Test with custom basis
        custom_basis = Basis(4, 3)
        result = expr.subs(custom_basis.sym_dict())
        
        # Verify: 4^3 - 2*4*3 + 3^4 - 7 = 64 - 24 + 81 - 7 = 114
        expected = 4**3 - 2*4*3 + 3**4 - 7
        assert result == expected


class TestUnpackingIntegration:
    """Test basis unpacking with legacy methods."""
    
    def test_basis_tuple_unpacking(self):
        """Test using basis.tuple() for positional unpacking."""
        p = P(281)
        basis = B.Collatz
        
        # Should be equivalent
        result1 = p.k(g=3, h=2)
        result2 = p.k(*basis.tuple())
        
        assert result1 == result2
    
    def test_basis_dict_unpacking(self):
        """Test using basis.dict() for keyword unpacking."""
        p = P(281)
        basis = B.Collatz
        
        # Should be equivalent
        result1 = p.k(g=3, h=2)
        result2 = p.k(**basis.dict())
        
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__])