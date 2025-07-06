"""
Functions module for functional programming with P-objects.

This module provides a comprehensive set of functional utilities for working with
P-class objects in Collatz analysis. It enables elegant, composable operations
through curried functions, filters, and data transformations.

The module follows functional programming principles to enable clean code like:
    P(p).cycle(filter=F.isodd, map=F.k(3,2))

Usage Patterns:
    >>> from plumial.utils import F
    >>> # Simple getters
    >>> F.n(p_obj)  # Get bit count
    >>> F.d(p_obj)  # Get difference polynomial
    >>> 
    >>> # Curried functions
    >>> k_func = F.k(g=3, h=2)  # Returns lambda p: p.k(3, 2)
    >>> values = [k_func(p) for p in cycle]
    >>> 
    >>> # Filters
    >>> odd_elements = [p for p in cycle if F.isodd(p)]
"""

from typing import Callable, Any, Optional, Iterator, Dict, List, Union
import pandas as pd

# Import type aliases for better type safety
from ..types import NumericType, OptionalNumeric, PValueProtocol


# Simple getter functions
def as_expr(p: PValueProtocol) -> Any:
    """Extract expression representation from P-object."""
    return p.d()  # Return difference polynomial expression


def as_poly(p: PValueProtocol) -> Any:
    """Extract polynomial representation from P-object."""
    return p.d()  # Return difference polynomial


def b(p: PValueProtocol, width: int = 0) -> str:
    """Extract binary string representation from P-object."""
    return p.b(width)


def d(p: PValueProtocol) -> Any:
    """Extract difference polynomial from P-object."""
    return p.d()


def e(p: PValueProtocol) -> int:
    """Extract even bit count from P-object."""
    return p.e()


def isodd(p: PValueProtocol) -> bool:
    """Check if p-value is odd."""
    return p.p() % 2 == 1


def ismin(p: PValueProtocol) -> bool:
    """Check if p-value is minimal in its cycle."""
    # Find minimum in cycle
    cycle_values = [obj.p() for obj in p.cycle()]
    return p.p() == min(cycle_values)


def isforced(p: PValueProtocol) -> bool:
    """Check if the cycle is forced by the p-value bit pattern."""
    return p.isforced()


def n(p: PValueProtocol) -> int:
    """Extract total bit count from P-object."""
    return p.n()


def o(p: PValueProtocol) -> int:
    """Extract odd bit count from P-object."""
    return p.o()


def p(p: PValueProtocol) -> int:
    """Extract p-value from P-object."""
    return p.p()


# Curried getter functions
def a(g: OptionalNumeric = None, h: OptionalNumeric = None) -> Callable[[PValueProtocol], Any]:
    """
    Return curried function for a-value extraction.
    
    Args:
        g: Optional g parameter for evaluation
        h: Optional h parameter for evaluation
        
    Returns:
        Lambda function that extracts a-value from P-object
        
    Examples:
        >>> a_func = F.a(g=3, h=2)
        >>> result = a_func(p_obj)  # Equivalent to p_obj.a(3, 2)
    """
    return lambda p_obj: p_obj.a(g, h)


def f(g: OptionalNumeric = None, h: OptionalNumeric = None) -> Callable[[PValueProtocol], Any]:
    """
    Return curried function for f-value extraction.
    
    Args:
        g: Optional g parameter for evaluation
        h: Optional h parameter for evaluation
        
    Returns:
        Lambda function that extracts f-value from P-object
    """
    return lambda p_obj: p_obj.f(g, h)


def k(g: OptionalNumeric = None, h: OptionalNumeric = None) -> Callable[[PValueProtocol], Any]:
    """
    Return curried function for k-value extraction.
    
    Args:
        g: Optional g parameter for evaluation
        h: Optional h parameter for evaluation
        
    Returns:
        Lambda function that extracts k-value from P-object
        
    Examples:
        >>> k_func = F.k(g=3, h=2)
        >>> values = [k_func(p) for p in cycle]
    """
    return lambda p_obj: p_obj.k(g, h)


def x(g: OptionalNumeric = None, h: OptionalNumeric = None) -> Callable[[PValueProtocol], Any]:
    """
    Return curried function for x-value extraction.
    
    Args:
        g: Optional g parameter for evaluation
        h: Optional h parameter for evaluation
        
    Returns:
        Lambda function that extracts x-value from P-object
    """
    return lambda p_obj: p_obj.x(g, h)


# Boolean filter functions
def isfalse(p: PValueProtocol) -> bool:
    """Always return False - useful as a filter."""
    return False


def istrue(p: PValueProtocol) -> bool:
    """Always return True - useful as a filter."""
    return True


def isnot(filter_func: Callable[[PValueProtocol], bool]) -> Callable[[PValueProtocol], bool]:
    """
    Return inverted filter function.
    
    Args:
        filter_func: Original filter function
        
    Returns:
        Inverted filter function
        
    Examples:
        >>> not_odd = F.isnot(F.isodd)
        >>> even_elements = [p for p in cycle if not_odd(p)]
    """
    return lambda p_obj: not filter_func(p_obj)


# Identity functions
def identity_map(p: PValueProtocol) -> PValueProtocol:
    """Identity mapping function - returns input unchanged."""
    return p


def identity_reducer(acc: Any, p: PValueProtocol) -> Any:
    """Identity reducer function - returns accumulator unchanged."""
    return acc


# Cycle navigation functions  
def next(p: PValueProtocol) -> PValueProtocol:
    """Get next element in cycle."""
    return p.next()


def pred(p: PValueProtocol) -> PValueProtocol:
    """Get predecessor element in cycle."""
    return p.pred()


# Data conversion functions
def to_object(p: PValueProtocol) -> Dict[str, Any]:
    """
    Convert P-object to dictionary representation.
    
    Args:
        p: P-object to convert
        
    Returns:
        Dictionary with p-object properties
        
    Examples:
        >>> obj_dict = F.to_object(p_obj)
        >>> print(obj_dict)  # {'p': 133, 'n': 7, 'o': 2, 'e': 5, ...}
    """
    return {
        'p': p.p(),
        'n': p.n(),
        'o': p.o(),
        'e': p.e(),
    }


def to_dataframe(iterator: Iterator[PValueProtocol], 
                g: OptionalNumeric = None, 
                h: OptionalNumeric = None) -> pd.DataFrame:
    """
    Convert iterator of P-objects to pandas DataFrame.
    
    Args:
        iterator: Iterator yielding P-objects
        g: Optional g parameter for polynomial evaluation
        h: Optional h parameter for polynomial evaluation
        
    Returns:
        Pandas DataFrame with P-object data
        
    Examples:
        >>> df = F.to_dataframe(P(133).cycle(), g=3, h=2)
        >>> print(df.head())
    """
    data = []
    for p_obj in iterator:
        row = {
            'p': p_obj.p(),
            'n': p_obj.n(),
            'o': p_obj.o(),
            'e': p_obj.e(),
        }
        
        # Add evaluated polynomials if g, h provided
        if g is not None and h is not None:
            try:
                row['d'] = p_obj.d(g, h)
                row['k'] = p_obj.k(g, h)
                row['f'] = p_obj.f(g, h)
                row['a'] = p_obj.a(g, h)
                row['x'] = p_obj.x(g, h)
            except Exception:
                # Handle cases where evaluation might fail
                pass
                
        data.append(row)
    
    return pd.DataFrame(data)


# Additional utility functions for Collatz analysis
def cycle_length(p: PValueProtocol) -> int:
    """Get the length of the cycle containing this P-object."""
    return p.n()


def cycle_min(p: PValueProtocol) -> int:
    """Get the minimum p-value in the cycle containing this P-object."""
    return min(obj.p() for obj in p.cycle())


def cycle_max(p: PValueProtocol) -> int:
    """Get the maximum p-value in the cycle containing this P-object."""
    return max(obj.p() for obj in p.cycle())


def is_cycle_start(p: PValueProtocol) -> bool:
    """Check if this P-object represents the start of its cycle (minimum p-value)."""
    return ismin(p)


# Functional composition utilities
def compose(*functions):
    """
    Compose multiple functions into a single function.
    
    Args:
        *functions: Functions to compose (applied right to left)
        
    Returns:
        Composed function
        
    Examples:
        >>> composed = F.compose(F.n, F.next_func)
        >>> result = composed(p_obj)  # n(next_func(p_obj))
    """
    def composed_function(x):
        result = x
        for func in reversed(functions):
            result = func(result)
        return result
    return composed_function


def map_filter(map_func: Callable = None, 
               filter_func: Callable = None):
    """
    Create a combined map-filter function for cycle processing.
    
    Args:
        map_func: Function to apply to each element
        filter_func: Function to filter elements
        
    Returns:
        Combined function for use in cycle operations
        
    Examples:
        >>> mf = F.map_filter(F.k(3,2), F.isodd)
        >>> # Use with P(p).cycle() method
    """
    if map_func is None:
        map_func = identity_map
    if filter_func is None:
        filter_func = istrue
        
    def combined(iterator):
        for item in iterator:
            if filter_func(item):
                yield map_func(item)
    
    return combined