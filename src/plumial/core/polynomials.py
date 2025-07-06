"""
Path object representations for Collatz conjecture analysis.

This module provides the core P class for working with path objects in
Collatz sequence analysis. Path objects are hydrated path identifiers (p-values)
that encode the binary path information of Collatz sequences, with support
for k polynomials, uv polynomials, d polynomials, and various
mathematical operations.

The module uses LRU caching for performance optimization and provides
comprehensive support for symbolic mathematics through SymPy integration.

Key Components:
    - _P: Internal polynomial class with mathematical operations
    - P(): Factory function with caching support
    - Cycle navigation methods (next, pred, cycle)
    - UV polynomial generation and transformations
    - Mathematical relationships: x*d = a*k, a = d/f, x = k/f

Examples:
    >>> from plumial import P
    >>> p = P(133)  # Create polynomial for p-value 133
    >>> print(p.n(), p.o(), p.e())  # Bit counts: n=7, o=2, e=5
    >>> uv_poly = p.uv()  # Get UV polynomial representation
    >>> k_val = p.k(g=3, h=2)  # Evaluate k polynomial at g=3, h=2
"""

import numpy as np
import sympy as sy
import math
from functools import lru_cache
from typing import Optional, Any, Iterator, Callable, Union

# Import type aliases for better type safety
from ..types import (
    NumericType, NumericOrSymbolic, OptionalNumeric, PolynomialTuple,
    GCDResult, CacheInfo
)
from .D import D
from ..utils.symbolic import g as g_sym, h as h_sym, u as u_sym, v as v_sym
from ..utils.matrix_utils import expr_as_vector, eat_minus_1


class _P:
    """
    Internal class representing a path polynomial in Collatz analysis.
    
    This class encapsulates the mathematical structure of a Collatz path polynomial,
    providing methods for polynomial evaluation, cycle navigation, and various
    mathematical transformations. It maintains cached expressions for performance
    and supports both symbolic and numerical operations.
    
    The class implements the mathematical relationships between different polynomial
    representations used in Collatz analysis, including UV polynomials, difference
    polynomials, and the fundamental relationship x*d = a*k.
    
    This class should not be instantiated directly. Use the P() factory function instead.
    
    Attributes:
        _p: The integer p-value this polynomial represents
        _d: Cached difference polynomial D(p)
        _next: Cached next element in the cycle
        _pred: Cached predecessor element in the cycle
        _expr_*: Cached symbolic expressions for various polynomial forms
    """
    
    def __init__(self, p: int, pred: Optional['_P'] = None, next: Optional['_P'] = None) -> None:
        """
        Initialize a path polynomial.
        
        Args:
            p: The p-value (must be a positive integer)
            pred: Optional predecessor P instance for cycle linking
            next: Optional next P instance for cycle linking
            
        Note:
            This constructor is internal. Use P() factory function instead.
        """
        self._p: int = p
        self._d: Optional[D] = None
        self._next: Optional['_P'] = next
        self._pred: Optional['_P'] = pred
        self._min: Optional[int] = None
        
        # Cached symbolic expressions
        self._expr_p: Optional[sy.Expr] = None
        self._expr_a: Optional[sy.Expr] = None
        self._expr_f: Optional[sy.Expr] = None
        self._expr_k: Optional[sy.Expr] = None
        self._expr_x: Optional[sy.Expr] = None
        self._expr_uv: Optional[sy.Expr] = None
        self._expr_gh: Optional[sy.Expr] = None
        
        # Initialize difference polynomial D(p)
        if self._d is None:
            self._d = D(p)
    
    def p(self) -> int:
        """Return the p-value."""
        return self._p
    
    def n(self) -> int:
        """Return the number of path bits."""
        return self._p.bit_length() - 1
    
    def o(self) -> int:
        """Return the number of odd bits."""
        return self._p.bit_count() - 1
    
    def e(self) -> int:
        """Return the number of even bits."""
        return self.n() - self.o()
    
    def d(self, g: OptionalNumeric = None, h: OptionalNumeric = None) -> NumericOrSymbolic:
        """
        Return the difference polynomial or its evaluation.
        
        Args:
            g: Value to substitute for g parameter (default: keep symbolic)
            h: Value to substitute for h parameter (default: keep symbolic)
            
        Returns:
            Difference polynomial h^e - g^o, either symbolic or evaluated
            
        Examples:
            >>> p = P(133)
            >>> p.d()  # Symbolic form
            h**5 - g**2
            >>> p.d(g=3, h=2)  # Numerical evaluation
            23
        """
        return self._d.d(g, h)
    
    def D(self):
        """Return the difference polynomial object."""
        return self._d
    
    def next(self) -> '_P':
        """Get the next element in the cycle."""
        if self._next is None:
            # Implement bit rotation for next p-value
            n = self.n()
            next_p = ((self._p - (1 << n)) | (self._p & 1) << n) >> 1 | (1 << n)
            self._next = P(next_p, pred=self)
        return self._next
    
    def pred(self) -> '_P':
        """Get the predecessor element in the cycle."""
        if self._pred is None:
            # Implement bit rotation for predecessor p-value
            n = self.n()
            stop = 1 << n
            mask = stop - 1
            pred_p = (self._p << 1) & mask | (self._p & mask) >> (n - 1) | stop
            self._pred = P(pred_p, next=self)
        return self._pred
    
    def cycle(self, map: Callable[['_P'], Any] = None, 
              filter: Callable[['_P'], bool] = None) -> Iterator['_P']:
        """Iterate through the cycle."""
        if map is None:
            map = lambda x: x
        if filter is None:
            filter = lambda x: True
            
        current = self
        for i in range(self.n()):
            if filter(current):
                yield map(current)
            current = current.next()
    
    def uv(self) -> sy.Expr:
        """
        Generate UV polynomial representation.
        
        This creates a polynomial in u and v variables that encodes the
        binary path structure of the p-value.
        
        Returns:
            SymPy expression in u and v variables
        """
        if self._expr_uv is None:
            u_ = self.o() - 1
            v_ = 0
            t = self.p()
            r = 0
            
            while t > 1:
                if t % 2 == 1:
                    r = r + (u_sym**u_) * (v_sym**v_)
                    u_ -= 1
                v_ += 1
                t >>= 1
                
            if isinstance(r, int):
                r = sy.sympify(r)
                
            self._expr_uv = r
            
        return self._expr_uv
    
    def k(self, g: Optional[Union[int, float]] = None, 
          h: Optional[Union[int, float]] = None) -> Union[sy.Expr, int, float]:
        """
        Calculate the k polynomial.
        
        The k polynomial is derived from the gh transformation of the uv polynomial.
        
        Args:
            g: Value to substitute for g (default: keep symbolic)
            h: Value to substitute for h (default: keep symbolic)
            
        Returns:
            k polynomial or its evaluation
        """
        if self._expr_k is None:
            # Calculate k from gh transformation of uv
            uv_expr = self.uv()
            # Transform u -> g*h, v -> h, then factor out h^(o-1)
            gh_expr = uv_expr.subs(u_sym, g_sym * h_sym).subs(v_sym, h_sym)
            if self.o() > 0:
                self._expr_k = (gh_expr / h_sym**(self.o() - 1)).factor()
            else:
                self._expr_k = gh_expr
        
        result = self._expr_k
        
        if g is not None:
            result = result.subs(g_sym, g)
        
        if h is not None:
            result = result.subs(h_sym, h)
            
        return result
    
    def ax(self) -> PolynomialTuple:
        """
        Calculate a pair of polynomials a,x such that x*d = a*k.
        
        This method computes the polynomial factorization that satisfies the
        fundamental relationship x*d = a*k, where d is the difference polynomial
        and k is the k polynomial. The factorization separates the difference
        polynomial ratio d/k into numerator (a) and denominator (x) components.
        
        Returns:
            Tuple of (a_polynomial, x_polynomial) where both are SymPy expressions
            
        Mathematical Relationship:
            x * d = a * k
            a = d / f (where f is the GCD factor)
            x = k / f
            
        Examples:
            >>> p = P(133)
            >>> a, x = p.ax()
            >>> # Verify the relationship symbolically
            >>> d_poly = p.d()
            >>> k_poly = p.k()
            >>> assert sy.expand(x * d_poly) == sy.expand(a * k_poly)
        """
        if self._expr_a is None:
            # Calculate as d/k, then separate into a,x using fraction
            fraction_expr = (self.d() / self.k()).factor(g_sym)
            a, x = sy.fraction(fraction_expr)
            
            # Move any leading negative coefficient into a term
            self._expr_a = eat_minus_1(a)
            self._expr_x = x
            
        return (self._expr_a, self._expr_x)
    
    def a(self, g: Optional[Union[int, float]] = None, 
          h: Optional[Union[int, float]] = None) -> Union[sy.Expr, int, float]:
        """
        Calculate the a coefficient.
        
        Args:
            g: Value to substitute for g (default: keep symbolic)
            h: Value to substitute for h (default: keep symbolic)
            
        Returns:
            a coefficient or its evaluation
        """
        if g is None or h is None:
            a, _ = self.ax()
            result = a
        else:
            # For numerical evaluation: a = d/f
            d_val = self.d(g, h)
            f_val = self.f(g, h)
            if f_val == 0:
                raise ZeroDivisionError("f is 0")
            result = d_val / f_val
            
        if g is not None and result != self._expr_a:
            result = result.subs(g_sym, g)
        if h is not None and result != self._expr_a:
            result = result.subs(h_sym, h)
            
        return result
    
    def x(self, g: Optional[Union[int, float]] = None, 
          h: Optional[Union[int, float]] = None) -> Union[sy.Expr, int, float]:
        """
        Calculate the x polynomial.
        
        Args:
            g: Value to substitute for g (default: keep symbolic)
            h: Value to substitute for h (default: keep symbolic)
            
        Returns:
            x polynomial or its evaluation
        """
        if g is None or h is None:
            _, x = self.ax()
            result = x
        else:
            # For numerical evaluation: x = k/f
            k_val = self.k(g, h)
            f_val = self.f(g, h)
            if f_val == 0:
                raise ZeroDivisionError("f is 0")
            result = k_val / f_val
            
        if g is not None and result != self._expr_x:
            result = result.subs(g_sym, g)
        if h is not None and result != self._expr_x:
            result = result.subs(h_sym, h)
            
        return result
    
    def f(self, g: OptionalNumeric = None, h: OptionalNumeric = None) -> GCDResult:
        """
        Calculate the f polynomial (GCD factor).
        
        The f polynomial represents the greatest common divisor factor that appears
        in the relationship between the difference polynomial d and the k polynomial.
        It is computed as the GCD of d and k, with special handling for even/odd
        values of g.
        
        Args:
            g: Value to substitute for g (default: keep symbolic)
            h: Value to substitute for h (default: keep symbolic)
            
        Returns:
            f polynomial (symbolic) or its numerical evaluation
            
        Mathematical Background:
            f = gcd(d, k) where d is the difference polynomial and k is the k polynomial
            For even g: f is computed over the entire cycle
            For odd g: f is the simple GCD of d and k
            
        Examples:
            >>> p = P(133)
            >>> f_symbolic = p.f()  # Symbolic form
            >>> f_numeric = p.f(g=3, h=2)  # Numerical evaluation
        """
        if self._expr_f is None:
            # Calculate as d/a factored
            a, _ = self.ax()
            self._expr_f = (self.d() / a).factor(g_sym).simplify()
        
        if g is None or h is None:
            result = self._expr_f
            if g is not None:
                result = result.subs(g_sym, g)
            if h is not None:
                result = result.subs(h_sym, h)
            return result
        
        # For full numerical evaluation, compute GCD directly
        k_val = self.k(g, h)
        d_val = self.d(g, h)
        
        if g % 2 == 0:
            # Even g case: compute GCD over cycle
            cycle_k_values = [p.k(g, h) for p in self.cycle()]
            f_val = math.gcd(*cycle_k_values, d_val)
        else:
            # Odd g case: simple GCD
            f_val = math.gcd(k_val, d_val)
            
        return f_val
    
    def isforced(self) -> bool:
        """
        Determine if the cycle is forced by the p-value bit pattern.
        
        A cycle is considered "forced" if the next operation cannot be determined 
        by inspecting the LSB bit of the x-value alone, but instead is determined 
        by the LSB bit of the p-value. This occurs when:
        
        1. The p-value contains any adjacent 1 bits in its binary representation, OR
        2. The top and bottom path bits are both 1
        
        This is a critical property for Collatz analysis: any counterexample to the 
        Collatz conjecture would have isforced() == False.
        
        Returns:
            True if the cycle is forced by the p-value bit pattern, False otherwise
            
        Mathematical Background:
            For unforced cycles: all(p.x(3,2) % 2 == p.p() % 2 for p in cycle)
            For forced cycles: the above condition is False for at least one element
            
        Examples:
            >>> P(9).isforced()   # Unforced cycle
            False
            >>> P(291).isforced() # Forced cycle  
            True
            >>> # Verify unforced property for P(9)
            >>> all(p.x(3,2) % 2 == p.p() % 2 for p in P(9).cycle())
            True
            >>> # Verify forced property for P(291)  
            >>> all(p.x(3,2) % 2 == p.p() % 2 for p in P(291).cycle())
            False
        """
        p_binary = self._p
        n = self.n()
        
        # Check for adjacent 1 bits in the path portion (excluding leading bit)
        # The path bits are bits 0 to n-1 (bit n is always 1)
        path_bits = p_binary & ((1 << n) - 1)
        
        # Check for adjacent 1s by shifting and ANDing
        # If any adjacent bits are both 1, this will be non-zero
        adjacent_ones = path_bits & (path_bits >> 1)
        if adjacent_ones != 0:
            return True
            
        # Check if top and bottom path bits are both 1
        # Bottom bit: bit 0
        # Top path bit: bit n-1 (the bit just below the leading 1)
        bottom_bit = (p_binary & 1) == 1
        top_path_bit = ((p_binary >> (n - 1)) & 1) == 1
        
        if bottom_bit and top_path_bit:
            return True
            
        return False
    
    def b(self, width: int = 0) -> str:
        """
        Return binary string representation of the p-value.
        
        Args:
            width: Minimum width (right-justified with zeros, default: 0)
            
        Returns:
            Binary string representation
            
        Examples:
            >>> p = P(133)
            >>> p.b()
            '10000101'
            >>> p.b(10)
            '0010000101'
        """
        from ..utils.binary import binary_string
        return binary_string(self._p, width)
    
    def __str__(self) -> str:
        return str(self._p)
    
    def __repr__(self) -> str:
        return str(self._p)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, _P):
            return self._p == other._p
        return False
    
    def __hash__(self) -> int:
        return hash(self._p)


@lru_cache(maxsize=1000)
def _create_p_cached(p: int) -> _P:
    """
    Create _P instance with LRU caching.
    
    This internal function provides caching for P instances to improve performance
    when the same p-values are accessed repeatedly. The cache uses LRU eviction
    and has a maximum size of 1000 entries.
    
    Args:
        p: The p-value (positive integer)
        
    Returns:
        Cached or newly created _P instance
        
    Note:
        This function is internal and should not be called directly.
        Use the P() factory function instead.
    """
    return _P(p)


def P(p, pred: Optional[_P] = None, next: Optional[_P] = None) -> _P:
    """
    Factory function for creating P instances with improved caching.
    
    This is the main entry point for creating path objects in the plumial
    library. It provides intelligent caching for performance while supporting
    cycle linking for predecessor/successor relationships.
    
    Args:
        p: The p-value (positive integer or binary string representing the path)
        pred: Optional predecessor P instance for cycle linking
        next: Optional next P instance for cycle linking
        
    Returns:
        _P instance, either from cache or newly created
        
    Raises:
        ValueError: If p <= 0 (p-values must be positive) or invalid binary string
        
    Examples:
        >>> p1 = P(133)  # Uses cache
        >>> p2 = P(133)  # Returns same instance from cache
        >>> assert p1 is p2
        >>> p3 = P('10000101')  # Binary string for 133
        >>> assert p1.p() == p3.p()
    """
    # Handle binary string input
    if isinstance(p, str):
        if not all(bit in '01' for bit in p):
            raise ValueError(f"Invalid binary string: {p}")
        if not p or p[0] != '1':
            raise ValueError(f"Binary string must start with '1': {p}")
        p = int(p, 2)
    
    if not isinstance(p, int) or p <= 0:
        raise ValueError(f"p-value must be a positive integer, got: {p}")
    
    if pred is None and next is None:
        # Use cached version for simple cases
        return _create_p_cached(p)
    else:
        # Handle pred/next relationships outside cache
        # This ensures we don't cache instances with specific relationships
        return _P(p, pred, next)


def clear_cache() -> None:
    """
    Clear the P instance cache.
    
    This function clears all cached P instances, which can be useful for
    memory management or when you want to ensure fresh instances are created.
    After calling this function, subsequent calls to P() will create new
    instances rather than returning cached ones.
    
    Examples:
        >>> p1 = P(133)
        >>> clear_cache()
        >>> p2 = P(133)
        >>> assert p1 is not p2  # Different instances after cache clear
    """
    _create_p_cached.cache_clear()


def cache_info() -> CacheInfo:
    """
    Get cache statistics for P instances.
    
    Returns cache hit/miss statistics and current cache size information
    for the P instance cache. This is useful for performance monitoring
    and cache effectiveness analysis.
    
    Returns:
        Cache information object with hits, misses, maxsize, and currsize
        
    Examples:
        >>> info = cache_info()
        >>> print(f"Cache hits: {info.hits}, misses: {info.misses}")
        >>> print(f"Cache size: {info.currsize}/{info.maxsize}")
    """
    return _create_p_cached.cache_info()