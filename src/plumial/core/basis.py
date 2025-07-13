"""
Basis and coordinate system objects for Plumial mathematical operations.

This module provides the foundation for encoding mathematical objects within
specific coordinate systems (bases) defined by (g,h) parameters.
"""

import sympy as sy
from typing import Dict, Tuple, Union, Any
from ..utils.symbolic import S


class Basis:
    """
    Represents a mathematical coordinate system defined by (g,h) parameters.
    
    A basis provides both semantic access methods (odd/even path parameters)
    and concrete access methods (g/h parameters) for different contexts.
    
    Examples:
        >>> # Create bases
        >>> symbolic_basis = Basis(S.g, S.h)
        >>> collatz_basis = Basis(3, 2)
        >>> 
        >>> # Semantic access
        >>> collatz_basis.odd()    # Returns 3 (odd-path parameter)
        >>> collatz_basis.even()   # Returns 2 (even-path parameter)
        >>> 
        >>> # Concrete access
        >>> collatz_basis.g()      # Returns 3 (g parameter)
        >>> collatz_basis.h()      # Returns 2 (h parameter)
        >>> 
        >>> # Unpacking for legacy methods
        >>> P(281).k(*collatz_basis.tuple())   # k(3, 2)
        >>> P(281).k(**collatz_basis.dict())   # k(g=3, h=2)
    """
    
    def __init__(self, g: Any, h: Any):
        """
        Initialize a basis with (g,h) parameters.
        
        Args:
            g: The g parameter (odd-path parameter)
            h: The h parameter (even-path parameter)
        """
        self._g = g
        self._h = h
    
    def odd(self) -> Any:
        """
        Return the odd-path parameter (g).
        
        This method provides semantic access to the parameter that governs
        transformations when odd bits are encountered in path polynomials.
        
        Returns:
            The g parameter
        """
        return self._g
    
    def even(self) -> Any:
        """
        Return the even-path parameter (h).
        
        This method provides semantic access to the parameter that governs
        transformations when even bits are encountered in path polynomials.
        
        Returns:
            The h parameter
        """
        return self._h
    
    def g(self) -> Any:
        """
        Return the g parameter (alias for odd()).
        
        This method provides concrete access to the g parameter for
        contexts where the traditional g,h notation is preferred.
        
        Returns:
            The g parameter
        """
        return self._g
    
    def h(self) -> Any:
        """
        Return the h parameter (alias for even()).
        
        This method provides concrete access to the h parameter for
        contexts where the traditional g,h notation is preferred.
        
        Returns:
            The h parameter
        """
        return self._h
    
    def is_concrete(self) -> Any:
        """
        Return true if the basis has concrete (numeric) axes.

        Returns:
            True if the axes are both numeric, false otherwise.
        """
        return sy.sympify(self._h).is_integer and sy.sympify(self._g).is_integer

    def tuple(self) -> Tuple[Any, Any]:
        """
        Return (g, h) tuple for positional unpacking.
        
        This enables usage like: method(*basis.tuple())
        
        Returns:
            Tuple of (g, h) parameters
        """
        return (self._g, self._h)
    
    def dict(self) -> Dict[str, Any]:
        """
        Return {'g': g, 'h': h} dict for keyword unpacking.
        
        This enables usage like: method(**basis.dict())
        
        Returns:
            Dictionary with 'g' and 'h' keys
        """
        return {'g': self._g, 'h': self._h}
    
    def sym_dict(self) -> Dict[Any, Any]:
        """
        Return {g_sym: g, h_sym: h} dict for symbolic substitution.
        
        This enables usage like: expression.subs(basis.sym_dict())
        
        Returns:
            Dictionary mapping symbolic variables to basis values
            
        Examples:
            >>> from plumial.utils.symbolic import g, h
            >>> basis = B.Collatz
            >>> expr = g**2 + h**3
            >>> result = expr.subs(basis.sym_dict())  # Evaluates with g=3, h=2
        """
        from ..utils.symbolic import g as g_sym, h as h_sym
        return {g_sym: self._g, h_sym: self._h}
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another basis.
        
        Two bases are equal if they have the same g and h parameters.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if bases are equal, False otherwise
        """
        if not isinstance(other, Basis):
            return False
        return self._g == other._g and self._h == other._h
    
    def __hash__(self) -> int:
        """
        Return hash of the basis for use in sets and dicts.
        
        Returns:
            Hash based on (g, h) parameters
        """
        return hash((self._g, self._h))
    
    def __repr__(self) -> str:
        """
        Return string representation of the basis.
        
        Returns:
            String representation showing g and h parameters
        """
        return f"Basis({self._g}, {self._h})"
    
    def __str__(self) -> str:
        """
        Return human-readable string representation.
        
        Returns:
            String representation for display
        """
        return f"Basis(g={self._g}, h={self._h})"


class B:
    """
    Namespace for predefined basis constants.
    
    This provides convenient access to commonly used mathematical bases
    for Collatz-like sequences and polynomial operations.
    
    Examples:
        >>> # Use predefined bases
        >>> P(281).encode(B.Collatz).k()
        >>> P(281).encode(B.Symbolic).k()
        >>> 
        >>> # Create custom basis
        >>> custom = Basis(5, 2)
        >>> P(281).encode(custom).k()
    """
    
    Symbolic = Basis(S.g, S.h)
    """Symbolic basis using symbolic g,h variables"""
    
    Collatz = Basis(3, 2)
    """Standard Collatz basis (3x+1, /2)"""
    
    Collatz_5_2 = Basis(5, 2)
    """5x+1 Collatz variant basis"""
    
    Collatz_7_2 = Basis(7, 2)
    """7x+1 Collatz variant basis"""
    
    Collatz_5_4 = Basis(5, 4)
    """5x+1, /4 Collatz variant basis"""
    
    Collatz_7_4 = Basis(7, 4)
    """7x+1, /4 Collatz variant basis"""


def resolve_basis(basis: Union[Basis, None] = None, 
                 g: Any = None, 
                 h: Any = None) -> Basis:
    """
    Resolve basis from various parameter combinations.
    
    This utility function handles the different ways a basis can be specified
    and returns a consistent Basis object.
    
    Args:
        basis: Explicit Basis object
        g: g parameter for custom basis
        h: h parameter for custom basis
        
    Returns:
        Resolved Basis object
        
    Raises:
        ValueError: If parameters are inconsistent or incomplete
        
    Examples:
        >>> resolve_basis(B.Collatz)           # Use existing basis
        >>> resolve_basis(g=3, h=2)           # Create from parameters
        >>> resolve_basis()                   # Returns symbolic basis
    """
    if basis is not None:
        if g is not None or h is not None:
            raise ValueError("Cannot specify both basis and g/h parameters")
        return basis
    
    if g is not None and h is not None:
        return Basis(g, h)
    
    if g is not None or h is not None:
        raise ValueError("Must specify both g and h parameters")
    
    # Default to symbolic basis
    return B.Symbolic