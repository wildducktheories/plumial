"""
D objects for Collatz conjecture analysis.

This module implements the D class representing d-polynomials of the form
d_p(g,h) = h^e - g^o, which are fundamental to the algebraic analysis of Collatz sequences.

The d-polynomial d_p(g,h) = h^e - g^o encodes the core mathematical structure of
Collatz sequences, where:
- h represents the "halving" operation (typically h=2)
- g represents the "3x+1" operation (typically g=3)
- e is the number of even steps (halving operations)
- o is the number of odd steps (3x+1 operations)

Key Features:
    - Symbolic representation and evaluation of d-polynomials
    - GCD computation for polynomial analysis
    - Factorization and algebraic manipulation
    - LRU caching for performance optimization
    - Support for both symbolic and numerical computations

Mathematical Background:
    The d-polynomial d_p(g,h) = h^e - g^o represents the net effect of a Collatz
    sequence segment.

Examples:
    >>> from plumial.core.D import D
    >>> d = D(133)  # Create from p-value 133
    >>> print(d.n(), d.o(), d.e())  # 7, 2, 5
    >>> poly = d.d()  # Get symbolic form: h^5 - g^2
    >>> collatz_d = d.encode(g=3, h=2)  # Encode for Collatz evaluation
    >>> result = collatz_d.d()  # Evaluate: 2^5 - 3^2 = 32 - 9 = 23
"""

import math
from functools import lru_cache
from typing import Any, List, Optional, Union

import numpy as np
import sympy as sy

# Import type aliases for better type safety
from ..types import (
    CacheInfo,
    FactorResult,
    GCDResult,
    NumericOrSymbolic,
    NumericType,
    OptionalNumeric,
)
from ..utils.symbolic import g as g_sym
from ..utils.symbolic import h as h_sym
from .basis import Basis, B, resolve_basis


class _D:
    """
    Internal class representing a d-polynomial d_p(g,h) = h^e - g^o.

    This class encapsulates the mathematical structure of a d-polynomial
    used in Collatz sequence analysis. It provides methods for symbolic manipulation
    and numerical evaluation.

    The class maintains the fundamental parameters n, o, e for the polynomial form h^e - g^o.

    This class should not be instantiated directly. Use the D() factory function instead.

    Mathematical Structure:
        - n: total number of bits (path length)
        - o: number of odd bits (3x+1 operations)
        - e: number of even bits (halving operations), where e = n - o
        - The polynomial form: h^e - g^o

    Attributes:
        _n: Total number of bits
        _o: Number of odd bits
        _e: Number of even bits (computed as n - o)
        _expr: Cached symbolic expression
    """

    def __init__(self, n: int, o: int, basis: Optional[Basis] = None) -> None:
        """
        Initialize d-polynomial from bit counts.

        Args:
            n: Total number of bits (must be >= 0)
            o: Number of odd bits (must be >= 0 and <= n)
            basis: The mathematical basis for this encoding (default: symbolic basis)

        Raises:
            ValueError: If n < 0, o < 0, or o > n

        Note:
            This constructor is internal. Use D() factory function instead.
        """
        self._n = n
        self._o = o
        self._e = n - o
        self._basis = basis if basis is not None else B.Symbolic

        # Cached symbolic expression
        self._expr = None

    def n(self) -> int:
        """Return total number of bits."""
        return self._n

    def o(self) -> int:
        """Return number of odd bits."""
        return self._o

    def e(self) -> int:
        """Return number of even bits."""
        return self._e


    def basis(self) -> Basis:
        """Return the basis of this encoding."""
        return self._basis

    def encode(
        self, 
        basis: Optional[Basis] = None, 
        g: OptionalNumeric = None, 
        h: OptionalNumeric = None
    ) -> "_D":
        """
        Create a new D object encoded in a different basis.
        
        This method enables the transitive encoding property where D objects
        can be transformed between different coordinate systems while preserving
        the underlying mathematical structure.
        
        Args:
            basis: Target basis for encoding
            g: g parameter for custom basis (alternative to basis parameter)
            h: h parameter for custom basis (alternative to basis parameter)
            
        Returns:
            New D object with same n,o values but different basis
            
        Examples:
            >>> d = D(133)                    # Symbolic basis
            >>> collatz_d = d.encode(B.Collatz)  # Collatz basis
            >>> custom_d = d.encode(g=5, h=2)    # Custom basis
            >>> back_d = collatz_d.encode()      # Back to symbolic basis
            >>> assert back_d == d              # Round-trip equality
        """
        # Handle empty encode() - return to symbolic basis
        if basis is None and g is None and h is None:
            return D(p=None, n=self._n, o=self._o, basis=B.Symbolic)
        
        # Resolve target basis
        target_basis = resolve_basis(basis=basis, g=g, h=h)
        
        # Return new D object with same n,o but different basis
        return D(p=None, n=self._n, o=self._o, basis=target_basis)

    def as_expr(self) -> sy.Expr:
        """
        Return the symbolic expression h^e - g^o.

        Returns:
            SymPy expression representing the d-polynomial
        """
        if self._expr is None:
            self._expr = h_sym**self._e - g_sym**self._o
        return self._expr

    def d(self) -> NumericOrSymbolic:
        """
        Evaluate the d-polynomial.

        Returns:
            Evaluated expression or symbolic form
            
        Examples:
            >>> d = D(133)
            >>> d.d()  # Symbolic form
            h**5 - g**2
            >>> collatz_d = D(133).encode(B.Collatz)
            >>> collatz_d.d()  # Uses basis automatically
            23
        """
        result = self.as_expr()

        # Use basis if it's not symbolic
        if self._basis != B.Symbolic:
            basis_dict = self._basis.dict()
            if basis_dict['g'] is not None:
                result = result.subs(g_sym, basis_dict['g'])
            if basis_dict['h'] is not None:
                result = result.subs(h_sym, basis_dict['h'])

        return result





    def __str__(self) -> str:
        """String representation showing the polynomial."""
        return str(self.as_expr())

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"D(n={self._n}, o={self._o}, e={self._e}): {self.as_expr()}"

    def __eq__(self, other) -> bool:
        """Equality comparison based on n, o, and basis values."""
        if isinstance(other, _D):
            return self._n == other._n and self._o == other._o and self._basis == other._basis
        return False

    def __hash__(self) -> int:
        """Hash based on n, o, and basis values."""
        return hash((self._n, self._o, self._basis))


@lru_cache(maxsize=1000)
def _create_d_cached(n: int, o: int, basis_hash: int) -> _D:
    """Create _D instance with LRU caching based on (n, o, basis) combination."""
    # Reconstruct basis from hash - this is a workaround since we can't cache
    # Basis objects directly. In practice, most usage will be with common bases.
    if basis_hash == hash(B.Symbolic):
        basis = B.Symbolic
    elif basis_hash == hash(B.Collatz):
        basis = B.Collatz
    elif basis_hash == hash(B.Collatz_5_2):
        basis = B.Collatz_5_2
    elif basis_hash == hash(B.Collatz_7_2):
        basis = B.Collatz_7_2
    elif basis_hash == hash(B.Collatz_5_4):
        basis = B.Collatz_5_4
    elif basis_hash == hash(B.Collatz_7_4):
        basis = B.Collatz_7_4
    else:
        # For custom bases, we skip caching to avoid hash collisions
        raise ValueError(f"Cannot reconstruct basis from hash {basis_hash}")
    
    return _D(n, o, basis=basis)


def D(p: Optional[int] = None, n: Optional[int] = None, o: Optional[int] = None, basis: Optional[Basis] = None) -> _D:
    """
    Factory function for creating D instances from p-values or bit counts.

    This is the main entry point for creating D objects. It supports multiple
    calling patterns for maximum flexibility while maintaining backward compatibility.

    Args:
        p: Integer p-value from which to extract n and o (must be positive)
        n: Total number of bits (alternative to p parameter)
        o: Number of odd bits (alternative to p parameter) 
        basis: The mathematical basis for this encoding (default: symbolic basis)

    Returns:
        _D instance representing the d-polynomial

    Raises:
        ValueError: If parameters are invalid or inconsistent

    Mathematical Background:
        The p-value encodes path information where:
        - n = bit_length(p) - 1 (total path bits)
        - o = bit_count(p) - 1 (odd path bits)
        - e = n - o (even path bits)
        - The d-polynomial is d_p(g,h) = h^e - g^o

    Examples:
        >>> d1 = D(133)                    # Creates from p-value (legacy)
        >>> d2 = D(133, basis=B.Collatz)   # Creates with specific basis
        >>> d3 = D(n=7, o=2)              # Creates from bit counts
        >>> d4 = D(n=7, o=2, basis=B.Collatz)  # Bit counts with basis
    """
    # Set default basis
    if basis is None:
        basis = B.Symbolic
    
    # Handle different calling patterns
    if p is not None:
        # Pattern: D(p) or D(p, basis=...)
        if n is not None or o is not None:
            raise ValueError("Cannot specify both p and n/o parameters")
        if p <= 0:
            raise ValueError("p-value must be positive")
        n = p.bit_length() - 1
        o = p.bit_count() - 1
    elif n is not None and o is not None:
        # Pattern: D(n=..., o=...) or D(n=..., o=..., basis=...)
        if n < 0 or o < 0 or o > n:
            raise ValueError(f"Invalid bit counts: n={n}, o={o}")
    else:
        raise ValueError("Must specify either p or both n and o parameters")

    # Use cached version for predefined bases
    try:
        return _create_d_cached(n, o, hash(basis))
    except ValueError:
        # Custom basis - skip caching
        return _D(n, o, basis=basis)


def clear_d_cache() -> None:
    """
    Clear the D instance cache.

    This function clears all cached D instances, which can be useful for
    memory management or when you want to ensure fresh instances are created.
    After calling this function, subsequent calls to D() will create new
    instances rather than returning cached ones.

    Examples:
        >>> d1 = D(133)
        >>> clear_d_cache()
        >>> d2 = D(133)
        >>> assert d1 is not d2  # Different instances after cache clear
    """
    _create_d_cached.cache_clear()


def d_cache_info() -> CacheInfo:
    """
    Get cache statistics for D instances.

    Returns cache hit/miss statistics and current cache size information
    for the D instance cache. This is useful for performance monitoring
    and cache effectiveness analysis.

    Returns:
        Cache information object with hits, misses, maxsize, and currsize

    Examples:
        >>> info = d_cache_info()
        >>> print(f"D cache hits: {info.hits}, misses: {info.misses}")
        >>> print(f"D cache size: {info.currsize}/{info.maxsize}")
    """
    return _create_d_cached.cache_info()
