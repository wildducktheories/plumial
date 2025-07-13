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
    >>> from plumial.core.basis import B
    >>> d = D(2, 5)  # Create from o=2, e=5 (h^5 - g^2)
    >>> print(d.o(), d.e(), d.n())  # 2, 5, 7
    >>> poly = d.d()  # Get symbolic form: h^5 - g^2
    >>> collatz_d = d.encode(B.Collatz)  # Encode for Collatz evaluation
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
        _o: Number of odd bits
        _e: Number of even bits
        _n: Total number of bits (computed as o + e)
        _expr: Cached symbolic expression
    """

    def __init__(self, o: int, e: int, basis: Optional[Basis] = None) -> None:
        """
        Initialize d-polynomial from odd and even bit counts.

        Args:
            o: Number of odd bits (must be >= 0)
            e: Number of even bits (must be >= 0)
            basis: The mathematical basis for this encoding (default: symbolic basis)

        Raises:
            ValueError: If o < 0 or e < 0

        Note:
            This constructor is internal. Use D() factory function instead.
        """
        if o < 0:
            raise ValueError(f"Number of odd bits must be >= 0, got o={o}")
        if e < 0:
            raise ValueError(f"Number of even bits must be >= 0, got e={e}")

        self._o = o
        self._e = e
        self._n = o + e
        self._basis = basis if basis is not None else B.Symbolic

        # Cached symbolic expression
        self._expr = None

    def n(self) -> int:
        """Return total number of bits (computed as o + e)."""
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

    def c(self) -> NumericOrSymbolic:
        """
        Calculate the ceiling of log_h(g).
        
        This method computes ceil(log_h(g)) where log_h(g) is the logarithm
        of g with base h. The result represents an important mathematical bound
        related to the d-polynomial structure.
        
        Returns:
            Ceiling value, either symbolic or evaluated based on the basis
            
        Mathematical Formula:
            c = ceil(log_h(g)) = ceil(log(g) / log(h))
            
        Examples:
            >>> d = D(2, 5)
            >>> d.c()  # Symbolic form
            >>> collatz_d = D(2, 5).encode(B.Collatz)
            >>> collatz_d.c()  # Numerical evaluation for g=3, h=2
        """
        # Get basis parameters
        basis_dict = self._basis.dict()
        g_val = basis_dict.get('g')
        h_val = basis_dict.get('h')
        
        if self._basis == B.Symbolic or g_val is None or h_val is None:
            # Return symbolic form
            return sy.ceiling(sy.log(g_sym, h_sym))
        else:
            # Return numerical evaluation
            if g_val <= 0 or h_val <= 0 or h_val == 1:
                raise ValueError(f"Invalid basis values for logarithm: g={g_val}, h={h_val}")
            
            log_val = math.log(g_val) / math.log(h_val)
            return math.ceil(log_val)

    def r(self) -> NumericOrSymbolic:
        """
        Calculate the remainder: c() * o() - e().
        
        This method computes the remainder value defined as the ceiling of log_h(g)
        times the number of odd bits minus the number of even bits.
        
        Returns:
            Remainder value, either symbolic or evaluated based on the basis
            
        Mathematical Formula:
            r = c * o - e = ceil(log_h(g)) * o - e
            
        Examples:
            >>> d = D(2, 5)
            >>> d.r()  # Symbolic form
            >>> collatz_d = D(2, 5).encode(B.Collatz)
            >>> collatz_d.r()  # Numerical evaluation: 2 * 2 - 5 = -1
        """
        c_val = self.c()
        
        if self._basis == B.Symbolic:
            # Return symbolic form
            return c_val * self._o - self._e
        else:
            # Return numerical evaluation
            return c_val * self._o - self._e

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
            >>> d = D(2, 5)                   # Symbolic basis
            >>> collatz_d = d.encode(B.Collatz)  # Collatz basis
            >>> custom_d = d.encode(g=5, h=2)    # Custom basis
            >>> back_d = collatz_d.encode()      # Back to symbolic basis
            >>> assert back_d == d              # Round-trip equality
        """
        # Handle empty encode() - return to symbolic basis
        if basis is None and g is None and h is None:
            return D(self._o, self._e, basis=B.Symbolic)
        
        # Resolve target basis
        target_basis = resolve_basis(basis=basis, g=g, h=h)
        
        # Return new D object with same o,e but different basis
        return D(self._o, self._e, basis=target_basis)

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
            >>> d = D(2, 5)
            >>> d.d()  # Symbolic form
            h**5 - g**2
            >>> collatz_d = D(2, 5).encode(B.Collatz)
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





    def G(self) -> sy.Matrix:
        """
        Create column vector of g powers from g^(o-1) down to g^0.
        
        This method generates a column vector containing powers of g in descending order,
        used for matrix operations with polynomial coefficients. The vector has o elements
        corresponding to the number of odd bits.
        
        Returns:
            SymPy Matrix column vector with g powers [g^(o-1), g^(o-2), ..., g^1, g^0]
            
        Mathematical Structure:
            For o odd bits, creates: [g^(o-1), g^(o-2), ..., g^1, g^0]^T
            
        Examples:
            >>> d = D(2, 5)  # o=2, e=5
            >>> g_vector = d.G()
            >>> # Returns Matrix([[g], [1]]) for powers g^1, g^0
            
        Matrix Operations:
            >>> d = D(3, 4)  # o=3, e=4  
            >>> G_vec = d.G()
            >>> # Returns Matrix([[g**2], [g], [1]]) for use in polynomial operations
        """
        if self._o == 0:
            # Special case: no odd bits means empty matrix
            return sy.Matrix([])
        
        # Create column vector with g powers from g^(o-1) down to g^0
        powers = [g_sym**i for i in range(self._o - 1, -1, -1)]
        return sy.Matrix(powers)

    def __str__(self) -> str:
        """String representation showing the polynomial."""
        return str(self.as_expr())

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"D(o={self._o}, e={self._e}): {self.as_expr()}"

    def __eq__(self, other) -> bool:
        """Equality comparison based on n, o, and basis values."""
        if isinstance(other, _D):
            return self._n == other._n and self._o == other._o and self._basis == other._basis
        return False

    def __hash__(self) -> int:
        """Hash based on n, o, and basis values."""
        return hash((self._n, self._o, self._basis))


@lru_cache(maxsize=1000)
def _create_d_cached(o: int, e: int, basis_hash: int) -> _D:
    """Create _D instance with LRU caching based on (o, e, basis) combination."""
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
    
    return _D(o, e, basis=basis)


def D(o: Optional[int] = None,
      e: Optional[int] = None,
      n: Optional[int] = None,
      basis: Optional[Basis] = None) -> _D:
    """
    Factory function for creating D instances from bit counts.

    This is the main entry point for creating D objects. It supports multiple
    calling patterns for flexibility.

    Args:
        o: Number of odd bits
        e: Number of even bits
        n: Total number of bits (alternative parameter)
        basis: The mathematical basis for this encoding (default: symbolic basis)

    Returns:
        _D instance representing the d-polynomial h^e - g^o

    Raises:
        ValueError: If parameters are invalid or inconsistent

    Usage Patterns:
        - D(o, e): Direct polynomial specification h^e - g^o
        - D(o=2, e=5): Named parameters
        - D(o=2, n=7): Named, compute e = n - o
        - D(e=5, n=7): Named, compute o = n - e

    Examples:
        >>> d1 = D(2, 5)                    # h^5 - g^2
        >>> d2 = D(o=2, e=5, basis=B.Collatz)  # With specific basis
        >>> d3 = D(o=2, n=7)               # Compute e = 7 - 2 = 5
        >>> d4 = D(e=5, n=7)               # Compute o = 7 - 5 = 2
    """
    # Set default basis
    if basis is None:
        basis = B.Symbolic
    
    # Count how many parameters are provided
    provided = sum(x is not None for x in [o, e, n])

    if provided == 2:
        if o is not None and e is not None:
            # D(o, e) or D(o=2, e=5) - direct specification
            pass  # o and e are already set
        elif o is not None and n is not None:
            # D(o=2, n=7) - compute e = n - o
            e = n - o
            if e < 0:
                raise ValueError(f"Invalid: n={n}, o={o} results in e={e} < 0")
        elif e is not None and n is not None:
            # D(e=5, n=7) - compute o = n - e
            o = n - e
            if o < 0:
                raise ValueError(f"Invalid: n={n}, e={e} results in o={o} < 0")
        else:
            raise ValueError("Must specify exactly two of (o, e, n)")
    elif provided == 3:
        # All three provided - validate consistency
        if n != o + e:
            raise ValueError(
                f"Inconsistent parameters: n={n} but o+e={o}+{e}={o+e}. "
                f"Specify exactly two of (o, e, n)."
            )
    else:
        raise ValueError("Must specify exactly two of (o, e, n)")

    # Use cached version for predefined bases
    try:
        return _create_d_cached(o, e, hash(basis))
    except ValueError:
        # Custom basis - skip caching
        return _D(o, e, basis=basis)


def clear_d_cache() -> None:
    """
    Clear the D instance cache.

    This function clears all cached D instances, which can be useful for
    memory management or when you want to ensure fresh instances are created.
    After calling this function, subsequent calls to D() will create new
    instances rather than returning cached ones.

    Examples:
        >>> d1 = D(2, 5)
        >>> clear_d_cache()
        >>> d2 = D(2, 5)
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
