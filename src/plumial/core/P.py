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
    >>> collatz_p = p.encode(g=3, h=2)  # Encode for Collatz evaluation
    >>> k_val = collatz_p.k()  # Evaluate k polynomial
"""

import math
from functools import lru_cache
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np
import sympy as sy

# Import type aliases for better type safety
from ..types import (
    CacheInfo,
    GCDResult,
    NumericOrSymbolic,
    NumericType,
    OptionalNumeric,
    PolynomialTuple,
)
from ..utils.matrix_utils import eat_minus_1, expr_as_vector
from ..utils.symbolic import g as g_sym
from ..utils.symbolic import h as h_sym
from ..utils.symbolic import u as u_sym
from ..utils.symbolic import v as v_sym
from .D import D
from .basis import Basis, B, resolve_basis


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
        _d: Cached D object D(p)
        _next: Cached next element in the cycle
        _pred: Cached predecessor element in the cycle
        _expr_*: Cached symbolic expressions for various polynomial forms
    """

    def __init__(
        self, 
        p: int, 
        basis: Optional[Basis] = None,
        pred: Optional["_P"] = None, 
        next: Optional["_P"] = None
    ) -> None:
        """
        Initialize a path polynomial.

        Args:
            p: The p-value (must be a positive integer)
            basis: The mathematical basis for this encoding (default: symbolic basis)
            pred: Optional predecessor P instance for cycle linking
            next: Optional next P instance for cycle linking

        Note:
            This constructor is internal. Use P() factory function instead.
        """
        self._p: int = p
        self._basis: Basis = basis if basis is not None else B.Symbolic
        self._d: Optional[D] = None
        self._next: Optional["_P"] = next
        self._pred: Optional["_P"] = pred
        self._min: Optional[int] = None

        # Cached symbolic expressions
        self._expr_p: Optional[sy.Expr] = None
        self._expr_a: Optional[sy.Expr] = None
        self._expr_f: Optional[sy.Expr] = None
        self._expr_k: Optional[sy.Expr] = None
        self._expr_x: Optional[sy.Expr] = None
        self._expr_uv: Optional[sy.Expr] = None
        self._expr_gh: Optional[sy.Expr] = None

        # Initialize D object with same basis
        if self._d is None:
            self._d = D(self.o(), self.e(), basis=self._basis)

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

    def c(self) -> Union[sy.Expr, int, float]:
        """
        Calculate the ceiling of log_h(g).
        
        Delegates to the D object's c() method.
        
        Returns:
            Ceiling value, either symbolic or evaluated based on the basis
        """
        return self._d.c()

    def r(self) -> Union[sy.Expr, int, float]:
        """
        Calculate the remainder: c() * o() - e().
        
        Delegates to the D object's r() method.
        
        Returns:
            Remainder value, either symbolic or evaluated based on the basis
        """
        return self._d.r()

    def basis(self) -> Basis:
        """Return the basis of this encoding."""
        return self._basis

    def encode(
        self, 
        basis: Optional[Basis] = None, 
        g: OptionalNumeric = None, 
        h: OptionalNumeric = None
    ) -> "_P":
        """
        Create a new P object encoded in a different basis.
        
        This method enables the transitive encoding property where P objects
        can be transformed between different coordinate systems while preserving
        the underlying p-value identity.
        
        Args:
            basis: Target basis for encoding
            g: g parameter for custom basis (alternative to basis parameter)
            h: h parameter for custom basis (alternative to basis parameter)
            
        Returns:
            New P object with same p-value but different basis
            
        Examples:
            >>> p = P(281)                    # Symbolic basis
            >>> collatz_p = p.encode(B.Collatz)  # Collatz basis
            >>> custom_p = p.encode(g=5, h=2)    # Custom basis
            >>> back_p = collatz_p.encode()      # Back to symbolic basis
            >>> assert back_p == p              # Round-trip equality
        """
        # Handle empty encode() - return to symbolic basis
        if basis is None and g is None and h is None:
            return P(self._p, basis=B.Symbolic)
        
        # Resolve target basis
        target_basis = resolve_basis(basis=basis, g=g, h=h)
        
        # Return new P object with same p-value but different basis
        return P(self._p, basis=target_basis)

    def d(self) -> NumericOrSymbolic:
        """
        Return the d-polynomial or its evaluation.

        Returns:
            Difference polynomial h^e - g^o, either symbolic or evaluated

        Examples:
            >>> p = P(133)
            >>> p.d()  # Symbolic form
            h**5 - g**2
            >>> collatz_p = P(133).encode(B.Collatz)
            >>> collatz_p.d()  # Uses basis automatically
            23
        """
        return self._d.d()

    def D(self):
        """Return the D object."""
        return self._d

    def next(self) -> "_P":
        """Get the next element in the cycle."""
        if self._next is None:
            # Implement bit rotation for next p-value
            n = self.n()
            next_p = ((self._p - (1 << n)) | (self._p & 1) << n) >> 1 | (1 << n)
            self._next = P(next_p, basis=self._basis, pred=self)
        return self._next

    def pred(self) -> "_P":
        """Get the predecessor element in the cycle."""
        if self._pred is None:
            # Implement bit rotation for predecessor p-value
            n = self.n()
            stop = 1 << n
            mask = stop - 1
            pred_p = (self._p << 1) & mask | (self._p & mask) >> (n - 1) | stop
            self._pred = P(pred_p, basis=self._basis, next=self)
        return self._pred

    def cycle(
        self, map: Callable[["_P"], Any] = None, filter: Callable[["_P"], bool] = None
    ) -> Iterator["_P"]:
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

    def k(self) -> Union[sy.Expr, int, float]:
        """
        Calculate the k polynomial.

        The k polynomial is derived from the gh transformation of the uv polynomial.

        Returns:
            k polynomial or its evaluation
            
        Examples:
            >>> p = P(133)
            >>> p.k()  # Symbolic form
            # Returns symbolic k polynomial
            >>> collatz_p = P(133).encode(B.Collatz)
            >>> collatz_p.k()  # Uses basis automatically
            # Returns numerical value using Collatz basis
        """
        if self._expr_k is None:
            # Calculate k from gh transformation of uv
            uv_expr = self.uv()
            # Transform u -> g*h, v -> h, then factor out h^(o-1)
            gh_expr = uv_expr.subs(u_sym, g_sym * h_sym).subs(v_sym, h_sym)
            if self.o() > 0:
                self._expr_k = (gh_expr / h_sym ** (self.o() - 1)).factor()
            else:
                self._expr_k = gh_expr

        result = self._expr_k

        # Use basis if it's not symbolic
        if self._basis != B.Symbolic:
            basis_dict = self._basis.dict()
            if basis_dict['g'] is not None:
                result = result.subs(g_sym, basis_dict['g'])
            if basis_dict['h'] is not None:
                result = result.subs(h_sym, basis_dict['h'])

        return result

    def ax(self) -> PolynomialTuple:
        """
        Calculate a pair of polynomials a,x such that x*d = a*k.

        This method computes the polynomial factorization that satisfies the
        fundamental relationship x*d = a*k, where d is the d-polynomial
        and k is the k polynomial. The factorization separates the d-polynomial
        ratio d/k into numerator (a) and denominator (x) components.

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

    def a(self) -> Union[sy.Expr, int, float]:
        """
        Calculate the a coefficient.

        Returns:
            a coefficient or its evaluation
        """
        a, _ = self.ax()
        result = a

        # Use basis if it's not symbolic
        if self._basis != B.Symbolic:
            basis_dict = self._basis.dict()
            if basis_dict['g'] is not None:
                result = result.subs(g_sym, basis_dict['g'])
            if basis_dict['h'] is not None:
                result = result.subs(h_sym, basis_dict['h'])

        return result

    def x(self) -> Union[sy.Expr, int, float]:
        """
        Calculate the x polynomial.

        Returns:
            x polynomial or its evaluation
        """
        _, x = self.ax()
        result = x

        # Use basis if it's not symbolic
        if self._basis != B.Symbolic:
            basis_dict = self._basis.dict()
            if basis_dict['g'] is not None:
                result = result.subs(g_sym, basis_dict['g'])
            if basis_dict['h'] is not None:
                result = result.subs(h_sym, basis_dict['h'])

        return result

    def f(self) -> GCDResult:
        """
        Calculate the f polynomial (GCD factor).

        The f polynomial represents the greatest common divisor factor that appears
        in the relationship between the d-polynomial d and the k polynomial.
        It is computed as the GCD of d and k, with special handling for even/odd
        values of g.

        Returns:
            f polynomial (symbolic) or its numerical evaluation

        Mathematical Background:
            f = gcd(d, k) where d is the d-polynomial and k is the k polynomial
            For even g: f is computed over the entire cycle
            For odd g: f is the simple GCD of d and k

        Examples:
            >>> p = P(133)
            >>> f_symbolic = p.f()  # Symbolic form
            >>> collatz_p = P(133).encode(B.Collatz)
            >>> f_numeric = collatz_p.f()  # Numerical evaluation
        """
        if self._expr_f is None:
            # Calculate as d/a factored
            a, _ = self.ax()
            self._expr_f = (self.d() / a).factor(g_sym).simplify()

        result = self._expr_f

        # Check if we have a concrete basis for numerical evaluation
        if self._basis != B.Symbolic:
            basis_dict = self._basis.dict()
            g_val = basis_dict.get('g')
            h_val = basis_dict.get('h')

            if g_val is not None and h_val is not None:
                # For full numerical evaluation, compute GCD directly
                k_val = self.k()
                d_val = self.d()

                if g_val % 2 == 0:
                    # Even g case: compute GCD over cycle
                    cycle_k_values = [p.k() for p in self.cycle()]
                    return math.gcd(*cycle_k_values, d_val)
                else:
                    # Odd g case: simple GCD
                    return math.gcd(k_val, d_val)
            else:
                # Substitute available parameters
                if g_val is not None:
                    result = result.subs(g_sym, g_val)
                if h_val is not None:
                    result = result.subs(h_sym, h_val)

        return result

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

    def G(self) -> sy.Matrix:
        """
        Create column vector of g powers from g^(o-1) down to g^0.
        
        This method delegates to the D object's G() method to generate a column vector
        containing powers of g in descending order, used for matrix operations with 
        polynomial coefficients.
        
        Returns:
            SymPy Matrix column vector with g powers [g^(o-1), g^(o-2), ..., g^1, g^0]
            
        Mathematical Structure:
            For o odd bits, creates: [g^(o-1), g^(o-2), ..., g^1, g^0]^T
            
        Examples:
            >>> p = P(133)  # o=2, e=5
            >>> g_vector = p.G()
            >>> # Returns Matrix([[g], [1]]) for powers g^1, g^0
            
        Matrix Operations:
            >>> p = P(281)  # o=3, e=5
            >>> G_vec = p.G()
            >>> # Returns Matrix([[g**2], [g], [1]]) for use in polynomial operations
            
        Note:
            This method delegates to the associated D object: self._d.G()
        """
        return self._d.G()

    def __str__(self) -> str:
        return str(self._p)

    def __repr__(self) -> str:
        return str(self._p)

    def __eq__(self, other) -> bool:
        if isinstance(other, _P):
            return self._p == other._p and self._basis == other._basis
        return False

    def __hash__(self) -> int:
        return hash((self._p, self._basis))


@lru_cache(maxsize=1000)
def _create_p_cached(p: int, basis_hash: int) -> _P:
    """
    Create _P instance with LRU caching based on (p-value, basis) combination.

    This internal function provides caching for P instances to improve performance
    when the same (p-value, basis) combinations are accessed repeatedly. The cache
    uses LRU eviction and has a maximum size of 1000 entries.

    Args:
        p: The p-value (positive integer)
        basis_hash: Hash of the basis for cache key

    Returns:
        Cached or newly created _P instance

    Note:
        This function is internal and should not be called directly.
        Use the P() factory function instead.
    """
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
        # This is a safety measure - custom bases won't be cached
        raise ValueError(f"Cannot reconstruct basis from hash {basis_hash}")
    
    return _P(p, basis=basis)


def P(p, basis: Optional[Basis] = None, pred: Optional[_P] = None, next: Optional[_P] = None) -> _P:
    """
    Factory function for creating P instances with improved caching.

    This is the main entry point for creating path objects in the plumial
    library. It provides intelligent caching for performance while supporting
    basis-aware encoding and cycle linking for predecessor/successor relationships.

    Args:
        p: The p-value (positive integer or binary string representing the path)
        basis: The mathematical basis for this encoding (default: symbolic basis)
        pred: Optional predecessor P instance for cycle linking
        next: Optional next P instance for cycle linking

    Returns:
        _P instance, either from cache or newly created

    Raises:
        ValueError: If p <= 0 (p-values must be positive) or invalid binary string

    Examples:
        >>> p1 = P(133)  # Uses symbolic basis and cache
        >>> p2 = P(133)  # Returns same instance from cache
        >>> assert p1 is p2
        >>> p3 = P('10000101')  # Binary string for 133
        >>> assert p1.p() == p3.p()
        >>> collatz_p = P(133, basis=B.Collatz)  # Different basis
        >>> assert collatz_p != p1  # Different basis means different object
    """
    # Handle binary string input
    if isinstance(p, str):
        if not all(bit in "01" for bit in p):
            raise ValueError(f"Invalid binary string: {p}")
        if not p or p[0] != "1":
            raise ValueError(f"Binary string must start with '1': {p}")
        p = int(p, 2)

    if not isinstance(p, int) or p <= 0:
        raise ValueError(f"p-value must be a positive integer, got: {p}")

    # Set default basis
    if basis is None:
        basis = B.Symbolic

    if pred is None and next is None:
        # Use cached version for simple cases with predefined bases
        try:
            return _create_p_cached(p, hash(basis))
        except ValueError:
            # Custom basis - skip caching
            return _P(p, basis=basis)
    else:
        # Handle pred/next relationships outside cache
        # This ensures we don't cache instances with specific relationships
        return _P(p, basis=basis, pred=pred, next=next)


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
