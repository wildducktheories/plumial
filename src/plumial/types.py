"""
Type definitions and aliases for the Plumial library.

This module provides common type aliases and protocols used throughout
the library for better type safety and code readability.
"""

from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, Union

import sympy as sy

# Type aliases for common complex types
NumericType = Union[int, float]
"""Type alias for numeric values (int or float)."""

SymbolicType = Union[sy.Symbol, sy.Expr]
"""Type alias for SymPy symbolic objects."""

NumericOrSymbolic = Union[sy.Expr, int, float, sy.Rational]
"""Type alias for values that can be either numeric or symbolic."""

OptionalNumeric = Optional[NumericType]
"""Type alias for optional numeric values."""

OptionalSymbolic = Optional[SymbolicType]
"""Type alias for optional symbolic values."""

BitPosition = Tuple[int, ...]
"""Type alias for bit position tuples."""

BitPositionIterator = Iterator[Tuple[int, ...]]
"""Type alias for iterators yielding bit position tuples."""

CycleIterator = Iterator[int]
"""Type alias for iterators yielding cycle values."""

SymbolDict = Dict[str, sy.Symbol]
"""Type alias for symbol lookup dictionaries."""

IndexedDict = Dict[str, sy.IndexedBase]
"""Type alias for indexed symbol lookup dictionaries."""

CoefficientsDict = Dict[int, sy.Expr]
"""Type alias for polynomial coefficient dictionaries."""

SubstitutionDict = Dict[SymbolicType, NumericOrSymbolic]
"""Type alias for symbolic substitution dictionaries."""

PowersDict = Dict[str, int]
"""Type alias for polynomial powers dictionaries."""

ExpressionList = List[sy.Expr]
"""Type alias for lists of symbolic expressions."""

CacheInfo = Any
"""Type alias for functools cache info objects."""


# Protocols for duck typing
class PValueProtocol(Protocol):
    """
    Protocol for objects that behave like P-values.

    Objects implementing this protocol must provide methods to access
    the p-value and its bit count properties.
    """

    def p(self) -> int:
        """Return the p-value as an integer."""
        ...

    def n(self) -> int:
        """Return the number of path bits (total bits - 1)."""
        ...


class DPolynomialProtocol(Protocol):
    """
    Protocol for objects that behave like D objects.

    Objects implementing this protocol must provide methods to access
    polynomial properties and evaluation.
    """

    def n(self) -> int:
        """Return the number of path bits."""
        ...

    def o(self) -> int:
        """Return the number of odd bits."""
        ...

    def e(self) -> int:
        """Return the number of even bits."""
        ...

    def d(
        self, g: OptionalNumeric = None, h: OptionalNumeric = None
    ) -> NumericOrSymbolic:
        """Evaluate the d-polynomial."""
        ...


class CacheableProtocol(Protocol):
    """
    Protocol for objects that support caching operations.

    Objects implementing this protocol must provide methods to
    manage cache state and retrieve cache information.
    """

    def cache_clear(self) -> None:
        """Clear the cache."""
        ...

    def cache_info(self) -> CacheInfo:
        """Return cache statistics."""
        ...


class MathematicalExpressionProtocol(Protocol):
    """
    Protocol for objects that can be used in mathematical expressions.

    This protocol defines the interface for objects that can participate
    in symbolic mathematical operations.
    """

    def __add__(self, other: Any) -> Any:
        """Addition operation."""
        ...

    def __sub__(self, other: Any) -> Any:
        """Subtraction operation."""
        ...

    def __mul__(self, other: Any) -> Any:
        """Multiplication operation."""
        ...

    def __pow__(self, other: Any) -> Any:
        """Exponentiation operation."""
        ...


# Type guards for runtime type checking
def is_numeric(value: Any) -> bool:
    """
    Check if a value is numeric (int or float).

    Args:
        value: Value to check

    Returns:
        True if value is int or float, False otherwise

    Examples:
        >>> is_numeric(42)
        True
        >>> is_numeric(3.14)
        True
        >>> is_numeric("not a number")
        False
    """
    return isinstance(value, (int, float))


def is_symbolic(value: Any) -> bool:
    """
    Check if a value is a SymPy symbolic object.

    Args:
        value: Value to check

    Returns:
        True if value is a SymPy symbol or expression, False otherwise

    Examples:
        >>> import sympy as sy
        >>> x = sy.Symbol('x')
        >>> is_symbolic(x)
        True
        >>> is_symbolic(42)
        False
    """
    return isinstance(value, (sy.Symbol, sy.Expr))


def is_p_value_like(obj: Any) -> bool:
    """
    Check if an object implements the PValueProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object has p() and n() methods, False otherwise

    Examples:
        >>> from plumial import P
        >>> p_obj = P(133)
        >>> is_p_value_like(p_obj)
        True
        >>> is_p_value_like(42)
        False
    """
    return (
        hasattr(obj, "p") and hasattr(obj, "n") and callable(obj.p) and callable(obj.n)
    )


def is_d_polynomial_like(obj: Any) -> bool:
    """
    Check if an object implements the DPolynomialProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object has required methods, False otherwise

    Examples:
        >>> from plumial.core.D import D
        >>> d_obj = D(2, 5)
        >>> is_d_polynomial_like(d_obj)
        True
        >>> is_d_polynomial_like("not a polynomial")
        False
    """
    required_methods = ["n", "o", "e", "d"]
    return all(
        hasattr(obj, method) and callable(getattr(obj, method))
        for method in required_methods
    )


# Constants for common mathematical values
DEFAULT_G_VALUE: int = 3
"""Default value for the g parameter in Collatz analysis."""

DEFAULT_H_VALUE: int = 2
"""Default value for the h parameter in Collatz analysis."""

MAX_CACHE_SIZE: int = 1000
"""Default maximum size for LRU caches."""

# Type hints for specific mathematical operations
PolynomialTuple = Tuple[sy.Expr, sy.Expr]
"""Type alias for polynomial pairs (e.g., a,x polynomials)."""

GCDResult = Union[sy.Expr, int, float]
"""Type alias for GCD computation results."""

FactorResult = sy.Expr
"""Type alias for polynomial factorization results."""

BinaryPattern = str
"""Type alias for binary pattern strings."""

CycleStatistics = Dict[str, Union[int, List[int]]]
"""Type alias for cycle statistics dictionaries."""

BitAnalysis = Dict[str, Union[str, int, List[int]]]
"""Type alias for bit pattern analysis results."""
