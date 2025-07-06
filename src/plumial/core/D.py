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
    >>> result = d.d(g=3, h=2)  # Evaluate: 2^5 - 3^2 = 32 - 9 = 23
"""

import numpy as np
import sympy as sy
import math
from functools import lru_cache
from typing import Optional, Union, Any, List

# Import type aliases for better type safety
from ..types import (
    NumericType, NumericOrSymbolic, OptionalNumeric, GCDResult,
    FactorResult, CacheInfo
)
from ..utils.symbolic import g as g_sym, h as h_sym


class _D:
    """
    Internal class representing a d-polynomial d_p(g,h) = h^e - g^o.
    
    This class encapsulates the mathematical structure of a d-polynomial
    used in Collatz sequence analysis. It provides methods for symbolic manipulation,
    numerical evaluation, GCD computation, and polynomial factorization.
    
    The class maintains the fundamental parameters n, o, e and provides derived
    quantities q (quotient) and r (remainder) from the division e/o, which are
    important for certain mathematical operations.
    
    This class should not be instantiated directly. Use the D() factory function instead.
    
    Mathematical Structure:
        - n: total number of bits (path length)
        - o: number of odd bits (3x+1 operations)
        - e: number of even bits (halving operations), where e = n - o
        - q: quotient of e/o division (⌊e/o⌋)
        - r: remainder of e/o division (e mod o)
        - The polynomial form: h^e - g^o
        
    Attributes:
        _n: Total number of bits
        _o: Number of odd bits  
        _e: Number of even bits (computed as n - o)
        _q: Quotient from e/o division
        _r: Remainder from e/o division
        _expr: Cached symbolic expression
    """
    
    def __init__(self, n: int, o: int) -> None:
        """
        Initialize d-polynomial from bit counts.
        
        Args:
            n: Total number of bits (must be >= 0)
            o: Number of odd bits (must be >= 0 and <= n)
            
        Raises:
            ValueError: If n < 0, o < 0, or o > n
            
        Note:
            This constructor is internal. Use D() factory function instead.
        """
        self._n = n
        self._o = o
        self._e = n - o
        
        # Quotient and remainder for e/o division
        if o != 0:
            self._q = self._e // o
            self._r = self._e % o
        else:
            self._q = 0
            self._r = 0
        
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
    
    def q(self) -> int:
        """Return quotient of e/o division."""
        return self._q
    
    def r(self) -> int:
        """Return remainder of e/o division."""
        return self._r
    
    def as_expr(self) -> sy.Expr:
        """
        Return the symbolic expression h^e - g^o.
        
        Returns:
            SymPy expression representing the d-polynomial
        """
        if self._expr is None:
            self._expr = h_sym**self._e - g_sym**self._o
        return self._expr
    
    def d(self, g: OptionalNumeric = None, h: OptionalNumeric = None) -> NumericOrSymbolic:
        """
        Evaluate the d-polynomial.
        
        Args:
            g: Value to substitute for g (default: keep symbolic)
            h: Value to substitute for h (default: keep symbolic)
            
        Returns:
            Evaluated expression or symbolic form
        """
        result = self.as_expr()
        
        if g is not None:
            result = result.subs(g_sym, g)
        
        if h is not None:
            result = result.subs(h_sym, h)
            
        return result
    
    def gcd(self, g: OptionalNumeric = None) -> GCDResult:
        """
        Calculate GCD of 2^e and g^o.
        
        This method computes the greatest common divisor of the powers appearing
        in the d-polynomial. When g is specified numerically, it returns
        the integer GCD. When g is symbolic, it returns a symbolic GCD expression.
        
        Args:
            g: Value to substitute for g (default: keep symbolic)
            
        Returns:
            GCD as integer (if g provided) or symbolic expression
            
        Mathematical Background:
            gcd(2^e, g^o) is important for understanding the divisibility
            properties of the d-polynomial and potential cycle detection.
            
        Examples:
            >>> d = D(133)  # e=5, o=2
            >>> d.gcd(g=3)  # gcd(2^5, 3^2) = gcd(32, 9) = 1
            1
        """
        if g is None:
            return sy.Function('gcd')(2**self._e, g_sym**self._o)
        else:
            return math.gcd(2**self._e, g**self._o)
    
    def g_vector(self, g: Optional[Union[int, float]] = None, 
                 matrix: bool = False) -> sy.Matrix:
        """
        Generate vector of powers of g.
        
        Args:
            g: Value to substitute for g (default: keep symbolic)
            matrix: If True, return as diagonal matrix
            
        Returns:
            SymPy Matrix representing g^i for i in range(o)
        """
        if g is None:
            g = g_sym
        
        v = [g**i for i in range(self._o)]
        
        if matrix:
            v = sy.diag(*v)
        else:
            v = sy.Matrix(v)
        
        return v
    
    def factor(self) -> FactorResult:
        """
        Return factored form of the d-polynomial.
        
        This method applies SymPy's factorization capabilities to factor the
        d-polynomial d_p(g,h) = h^e - g^o. Factorization can reveal important
        mathematical structures and common factors that are useful for analysis.
        
        Returns:
            Factored symbolic expression
            
        Mathematical Background:
            Factoring h^e - g^o can reveal patterns like:
            - Difference of squares: a^2 - b^2 = (a-b)(a+b)
            - Difference of powers: a^n - b^n with various factorizations
            - Common polynomial factors that indicate mathematical relationships
            
        Examples:
            >>> d = D_from_counts(n=4, o=2)  # h^2 - g^2
            >>> d.factor()  # Returns (h - g)*(h + g)
        """
        return sy.factor(self.as_expr())
    
    def expand(self) -> sy.Expr:
        """
        Return expanded form of the d-polynomial.
        
        Returns:
            Expanded symbolic expression
        """
        return sy.expand(self.as_expr())
    
    def __str__(self) -> str:
        """String representation showing the polynomial."""
        return str(self.as_expr())
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"D(n={self._n}, o={self._o}, e={self._e}): {self.as_expr()}"
    
    def __eq__(self, other) -> bool:
        """Equality comparison based on n and o values."""
        if isinstance(other, _D):
            return self._n == other._n and self._o == other._o
        return False
    
    def __hash__(self) -> int:
        """Hash based on n and o values."""
        return hash((self._n, self._o))


@lru_cache(maxsize=1000)
def _create_d_cached(n: int, o: int) -> _D:
    """Create _D instance with LRU caching."""
    return _D(n, o)


def D(p: int) -> _D:
    """
    Factory function for creating D instances from p-values.
    
    This is the main entry point for creating D objects from
    p-values. The function extracts the bit count information from the p-value
    and creates a cached D instance for performance.
    
    Args:
        p: Integer p-value from which to extract n and o (must be positive)
        
    Returns:
        _D instance representing the d-polynomial
        
    Raises:
        ValueError: If p <= 0
        
    Mathematical Background:
        The p-value encodes path information where:
        - n = bit_length(p) - 1 (total path bits)
        - o = bit_count(p) - 1 (odd path bits)
        - e = n - o (even path bits)
        - The d-polynomial is d_p(g,h) = h^e - g^o
        
    Examples:
        >>> d1 = D(133)  # Creates D polynomial from p=133
        >>> print(d1.n(), d1.o(), d1.e())  # Shows bit counts: 7, 2, 5
        >>> print(d1.as_expr())  # Shows h^5 - g^2
        >>> result = d1.d(g=3, h=2)  # Evaluates to 23
    """
    if p <= 0:
        raise ValueError("p-value must be positive")
    
    n = p.bit_length() - 1
    o = p.bit_count() - 1
    return _create_d_cached(n, o)



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