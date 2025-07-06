"""
Symbolic variable definitions for Collatz conjecture analysis.

This module provides centralized symbolic variables used throughout
the mathematical computations. It maintains compatibility with the original
collatz library while providing better organization and documentation.
"""

import sympy as sy
from typing import List, Dict, Any

# Primary symbolic variables for gh-polynomial forms
g = sy.Symbol('g')  # Generator parameter (typically g=3)
h = sy.Symbol('h')  # Base parameter (typically h=2)

# Polynomial and mathematical variables
a = sy.Symbol('a')  # Additive constant parameter
d = sy.Symbol('d')  # d-polynomial
k = sy.Symbol('k')  # Polynomial coefficient
x = sy.Symbol('x')  # Variable
f = sy.Symbol('f')  # Function or factor

# UV polynomial variables (for path representation)
u = sy.Symbol('u')  # U parameter for UV polynomials
v = sy.Symbol('v')  # V parameter for UV polynomials

# Bit count and index variables
m = sy.Symbol('m')  # Generic counter/multiplier
n = sy.Symbol('n')  # Total number of bits (path length)
o = sy.Symbol('o')  # Number of odd bits
e = sy.Symbol('e')  # Number of even bits

# Index and loop variables
i = sy.Symbol('i')  # Loop index variable
j = sy.Symbol('j')  # Secondary loop index variable

# P-value and related variables
p = sy.Symbol('p')  # P-value symbol

# Additional mathematical variables
b = sy.Symbol('b')  # Binary/bit variable
c = sy.Symbol('c')  # Coefficient variable
r = sy.Symbol('r')  # Remainder variable
q = sy.Symbol('q')  # Quotient variable

# Mathematical constants
two = sy.Integer(2)    # Constant 2
three = sy.Integer(3)  # Constant 3

# Common substitution dictionaries for evaluation
COLLATZ_3_2 = {g: 3, h: 2}      # Standard Collatz (3x+1, /2)
COLLATZ_5_2 = {g: 5, h: 2}      # 5x+1 variant
COLLATZ_7_2 = {g: 7, h: 2}      # 7x+1 variant  
COLLATZ_15_2 = {g: 15, h: 2}    # 15x+1 variant
COLLATZ_8_3 = {g: 8, h: 3}      # 8x+1, /3 variant
COLLATZ_5_3 = {g: 5, h: 3}      # 5x+1, /3 variant

# Standard alias
COLLATZ_STD = COLLATZ_3_2        # Standard Collatz parameters

# Transformation constants for u,v ↔ g,h conversions
TRANSFORM_UV_TO_GH = {u: g*h, v: h}      # u→gh, v→h transformation
TRANSFORM_GH_TO_UV = {g: u/v, h: v}      # g→u/v, h→v transformation

# Indexed symbols for sequences and arrays
p_indexed = sy.IndexedBase('p')  # Indexed p-values: p[i]
x_indexed = sy.IndexedBase('x')  # Indexed x-values: x[i]
k_indexed = sy.IndexedBase('k')  # Indexed k-values: k[i]
b_indexed = sy.IndexedBase('b')  # Indexed b-values: b[i]
e_indexed = sy.IndexedBase('e')  # Indexed e-values: e[i]
o_indexed = sy.IndexedBase('o')  # Indexed o-values: o[i]
m_indexed = sy.IndexedBase('m')  # Indexed m-values: m[i]

# Collections for convenience and organization
primary_symbols = [g, h]
polynomial_symbols = [a, d, k, x, f]
uv_symbols = [u, v]
bit_count_symbols = [m, n, o, e]
index_symbols = [i, j]
additional_symbols = [p, b, c, r, q]
constants = [two, three]
indexed_symbols = [p_indexed, x_indexed, k_indexed, b_indexed, e_indexed, o_indexed, m_indexed]

# All basic symbols (non-indexed)
all_symbols = (primary_symbols + polynomial_symbols + uv_symbols + 
               bit_count_symbols + index_symbols + additional_symbols)

# Symbol lookup dictionary for dynamic access
symbol_dict: Dict[str, sy.Symbol] = {
    'g': g, 'h': h, 'a': a, 'd': d, 'k': k, 'x': x, 'f': f,
    'u': u, 'v': v, 'm': m, 'n': n, 'o': o, 'e': e, 'i': i, 'j': j,
    'p': p, 'b': b, 'c': c, 'r': r, 'q': q, 'two': two, 'three': three
}

# Substitution constants lookup
substitution_dict = {
    'COLLATZ_3_2': COLLATZ_3_2,
    'COLLATZ_5_2': COLLATZ_5_2, 
    'COLLATZ_7_2': COLLATZ_7_2,
    'COLLATZ_15_2': COLLATZ_15_2,
    'COLLATZ_8_3': COLLATZ_8_3,
    'COLLATZ_5_3': COLLATZ_5_3,
    'COLLATZ_STD': COLLATZ_STD,
    'TRANSFORM_UV_TO_GH': TRANSFORM_UV_TO_GH,
    'TRANSFORM_GH_TO_UV': TRANSFORM_GH_TO_UV
}

# Indexed symbol lookup
indexed_dict: Dict[str, sy.IndexedBase] = {
    'p': p_indexed, 'x': x_indexed, 'k': k_indexed, 'b': b_indexed,
    'e': e_indexed, 'o': o_indexed, 'm': m_indexed
}


def get_symbol(name: str) -> sy.Symbol:
    """
    Get a symbol by name.
    
    Args:
        name: Symbol name (e.g., 'g', 'h', 'u', 'v')
        
    Returns:
        SymPy symbol
        
    Raises:
        KeyError: If symbol name is not found
        
    Examples:
        >>> get_symbol('g')
        g
        >>> get_symbol('h')
        h
    """
    if name in symbol_dict:
        return symbol_dict[name]
    raise KeyError(f"Symbol '{name}' not found. Available symbols: {list(symbol_dict.keys())}")


def get_indexed_symbol(name: str) -> sy.IndexedBase:
    """
    Get an indexed symbol by name.
    
    Args:
        name: Base symbol name (e.g., 'p', 'x', 'k')
        
    Returns:
        SymPy indexed base
        
    Raises:
        KeyError: If indexed symbol name is not found
        
    Examples:
        >>> get_indexed_symbol('p')
        p
        >>> get_indexed_symbol('p')[0]
        p[0]
    """
    if name in indexed_dict:
        return indexed_dict[name]
    raise KeyError(f"Indexed symbol '{name}' not found. Available: {list(indexed_dict.keys())}")


def substitute_values(expr: sy.Expr, **kwargs) -> sy.Expr:
    """
    Substitute values into an expression using symbol names.
    
    Args:
        expr: SymPy expression
        **kwargs: Symbol names and their values (e.g., g=3, h=2)
        
    Returns:
        Expression with substituted values
        
    Examples:
        >>> expr = g**2 + h
        >>> substitute_values(expr, g=3, h=2)
        11
    """
    substitutions = {}
    for name, value in kwargs.items():
        if name in symbol_dict:
            substitutions[symbol_dict[name]] = value
        else:
            raise KeyError(f"Unknown symbol name: {name}")
    
    return expr.subs(substitutions)


def create_polynomial_expression(powers: Dict[str, int]) -> sy.Expr:
    """
    Create a polynomial expression from symbol powers.
    
    Args:
        powers: Dictionary mapping symbol names to their powers
        
    Returns:
        SymPy polynomial expression
        
    Examples:
        >>> create_polynomial_expression({'g': 2, 'h': 3})
        g**2*h**3
        >>> create_polynomial_expression({'u': 1, 'v': 2})
        u*v**2
    """
    expr = sy.Integer(1)
    for name, power in powers.items():
        if name in symbol_dict:
            if power != 0:
                expr *= symbol_dict[name] ** power
        else:
            raise KeyError(f"Unknown symbol name: {name}")
    
    return expr

def uv_to_gh_transform(uv_expr: sy.Expr) -> sy.Expr:
    """
    Transform u,v polynomial to g,h form using TRANSFORM_UV_TO_GH.
    
    This implements the substitution u → g·h, v → h which transforms
    sigma polynomials to k polynomials (after appropriate scaling).
    
    Args:
        uv_expr: Expression in u and v variables
        
    Returns:
        Expression in g and h variables
        
    Examples:
        >>> uv_expr = u**2 + v
        >>> uv_to_gh_transform(uv_expr)
        g**2*h**2 + h
        >>> # Using the constant directly:
        >>> uv_expr.subs(TRANSFORM_UV_TO_GH)
        g**2*h**2 + h
    """
    return uv_expr.subs(TRANSFORM_UV_TO_GH)


def gh_to_uv_transform(gh_expr: sy.Expr) -> sy.Expr:
    """
    Transform g,h polynomial to u,v form using TRANSFORM_GH_TO_UV.
    
    This implements the reverse substitution g → u/v, h → v which transforms
    k polynomials back to sigma polynomials (with appropriate scaling).
    
    Args:
        gh_expr: Expression in g and h variables
        
    Returns:
        Expression in u and v variables
        
    Examples:
        >>> # For expressions that result from u,v → g,h transformation:
        >>> gh_expr = g*h + h**2  # This came from u + v**2 originally
        >>> gh_to_uv_transform(gh_expr)
        u + v**2
        >>> # Using the constant directly:
        >>> gh_expr.subs(TRANSFORM_GH_TO_UV)
        u + v**2
    """
    return gh_expr.subs(TRANSFORM_GH_TO_UV)


def extract_coefficients(expr: sy.Expr, variable: sy.Symbol) -> Dict[int, sy.Expr]:
    """
    Extract coefficients of powers of a variable from an expression.
    
    Args:
        expr: SymPy expression
        variable: Variable to extract coefficients for
        
    Returns:
        Dictionary mapping powers to coefficients
        
    Examples:
        >>> expr = 3*g**2 + 2*g + 5
        >>> extract_coefficients(expr, g)
        {0: 5, 1: 2, 2: 3}
    """
    poly = sy.Poly(expr, variable)
    coeffs = poly.all_coeffs()
    degree = poly.degree()
    
    # Create dictionary mapping power to coefficient
    result = {}
    for i, coeff in enumerate(coeffs):
        power = degree - i
        result[power] = coeff
    
    return result


# Compatibility layer for original library usage patterns
class SymbolNamespace:
    """
    Namespace class to mimic the original symbols module usage pattern.
    
    This allows code like: S.g, S.h, S.u, S.v
    where S = SymbolNamespace()
    """
    
    def __init__(self):
        # Add all symbols as attributes
        for name, symbol in symbol_dict.items():
            setattr(self, name, symbol)
        
        # Add indexed symbols
        for name, symbol in indexed_dict.items():
            setattr(self, f"{name}_indexed", symbol)
    
    def __getattr__(self, name: str):
        # Fallback for any missing attributes
        if name in symbol_dict:
            return symbol_dict[name]
        elif name in indexed_dict:
            return indexed_dict[name]
        else:
            raise AttributeError(f"Symbol '{name}' not found")


# Indexed symbols namespace class
class IndexedNamespace:
    """
    Namespace class for indexed symbols to mimic the original I module usage pattern.
    
    This allows code like: I.p[0], I.x[i], I.k[i,j]
    where I = IndexedNamespace()
    """
    
    def __init__(self):
        # Add all indexed symbols as attributes
        for name, symbol in indexed_dict.items():
            setattr(self, name, symbol)
    
    def __getattr__(self, name: str):
        # Fallback for any missing attributes
        if name in indexed_dict:
            return indexed_dict[name]
        else:
            raise AttributeError(f"Indexed symbol '{name}' not found")


# Import functions module for F namespace
from . import functions as _functions_module

# Functions namespace class
class FunctionsNamespace:
    """
    Namespace class for functions to mimic the original F module usage pattern.
    
    This allows code like: F.isodd(p), F.k(3,2), F.next_func(p)
    where F = FunctionsNamespace()
    """
    
    def __init__(self):
        # Add all functions from the functions module as attributes
        for name in dir(_functions_module):
            if not name.startswith('_'):
                attr = getattr(_functions_module, name)
                if callable(attr):
                    setattr(self, name, attr)
    
    def __getattr__(self, name: str):
        # Fallback for any missing attributes
        if hasattr(_functions_module, name):
            return getattr(_functions_module, name)
        else:
            raise AttributeError(f"Function '{name}' not found")


# Create default namespace instances for compatibility
S = SymbolNamespace()
I = IndexedNamespace()
F = FunctionsNamespace()