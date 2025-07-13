"""
Plumial: A Python package for Collatz conjecture analysis and computation.

This package provides comprehensive tools and mathematical frameworks for analyzing
the Collatz conjecture through polynomial representations, d-polynomials,
and binary path analysis.

Key Features:
    - Path Polynomials (P class): Represent Collatz sequence paths as mathematical polynomials
    - D Objects (D class): Analyze d-polynomials d_p(g,h) = h^e - g^o fundamental to Collatz analysis
    - Binary Operations: Comprehensive bit-level analysis and cycle navigation
    - Symbolic Mathematics: Full SymPy integration for symbolic computation
    - High Performance: LRU caching and optimized algorithms
    - Type Safety: Comprehensive type hints and modern Python practices

Core Classes:
    P: path object class for sequence analysis
        - uv polynomial representations
        - Mathematical relationships (x*d = a*k)
        - Cycle navigation and analysis

    D: d-polynomial class for algebraic analysis
        - Symbolic form h^e - g^o
        - GCD computation and factorization
        - Polynomial evaluation and manipulation

Mathematical Background:
    The Collatz conjecture studies sequences defined by:
    - If n is even: n → n/2
    - If n is odd: n → 3n+1

    This package encodes these operations using polynomial representations where:
    - h represents halving operations (typically h=2)
    - g represents the 3x+1 operation (typically g=3)
    - Path polynomials encode the sequence of operations
    - Difference polynomials h^e - g^o capture cycle behavior

Examples:
    >>> from plumial import P, D
    >>>
    >>> # Create a path polynomial for p-value 133
    >>> p = P(133)
    >>> print(f"Path bits: n={p.n()}, odd={p.o()}, even={p.e()}")
    >>>
    >>> # Get uv polynomial representation
    >>> uv_poly = p.uv()
    >>> print(f"uv polynomial: {uv_poly}")
    >>>
    >>> # Evaluate k polynomial using modern encoding approach
    >>> collatz_p = p.encode(g=3, h=2)
    >>> k_value = collatz_p.k()
    >>> print(f"k(3,2) = {k_value}")
    >>>
    >>> # Create d-polynomial
    >>> d = D(2, 5)
    >>> print(f"Difference polynomial: {d.d()}")  # h^5 - g^2
    >>> collatz_d = d.encode(g=3, h=2)
    >>> print(f"Evaluated at g=3, h=2: {collatz_d.d()}")  # 23

Installation:
    pip install plumial

Dependencies:
    - sympy: Symbolic mathematics
    - numpy: Numerical computing
    - typing: Type hints (Python 3.8+)

Author: Jon Seymour <jon@wildducktheories.com>
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Jon Seymour"
__email__ = "jon@wildducktheories.com"

from .core import D, P

__all__ = [
    "__version__",
    "P",
    "D",
]
