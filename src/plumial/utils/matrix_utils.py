"""
Matrix and vector utilities for polynomial manipulation.

This module provides utility functions for converting between symbolic expressions
and matrix/vector representations used in Collatz analysis.
"""

from typing import Optional, Union

import numpy as np
import sympy as sy

from .symbolic import g as g_sym


def expr_as_vector(
    expr: sy.Expr, dim: Optional[int] = None, symbol: sy.Symbol = g_sym
) -> sy.Matrix:
    """
    Convert a polynomial expression to a vector of coefficients.

    Args:
        expr: Symbolic expression to convert
        dim: Desired dimension (pads with zeros if needed)
        symbol: Symbol to extract coefficients for

    Returns:
        SymPy Matrix representing coefficient vector
    """
    if expr.is_number:
        coeffs = [expr]
    else:
        poly = sy.Poly(expr, symbol)
        coeffs = poly.all_coeffs()

    if dim is None:
        dim = len(coeffs)

    # Reverse to get ascending powers
    coeffs.reverse()

    # Pad with zeros if needed
    coeffs.extend([0] * (dim - len(coeffs)))

    return sy.Matrix(coeffs)


def vector_as_expr(
    v: sy.Matrix, expand: bool = True, symbol: sy.Symbol = g_sym
) -> sy.Expr:
    """
    Convert a coefficient vector back to a polynomial expression.

    Args:
        v: Vector of coefficients
        expand: Whether to expand the result
        symbol: Symbol for the polynomial variable

    Returns:
        Symbolic expression
    """
    if len(v) > 1:
        result = sy.sympify(v[0]) + symbol * vector_as_expr(
            v[1:], expand=False, symbol=symbol
        )
    elif len(v) == 1:
        result = v[0]
    else:
        raise ValueError("empty vector")

    if expand:
        result = result.expand()

    return result


def eat_minus_1(expr: sy.Expr, check: bool = True) -> sy.Expr:
    """
    Normalize expression to move leading negative coefficients into terms.

    This function ensures expressions like -1*(g-4)*(4+g) are rendered as (4-g)*(4+g).

    Args:
        expr: Expression to normalize
        check: Whether to verify the result matches the original

    Returns:
        Normalized expression
    """
    if not isinstance(expr, sy.Mul):
        return expr

    a, b = expr.as_two_terms()
    result = None

    if a == sy.zoo:
        return expr

    if a.is_number and a < 0:
        # Try to absorb negative coefficient into subsequent terms
        result = -a * eat_minus_1(b, check=False)
    elif isinstance(a, sy.Add):
        c, d = a.as_two_terms()

        swap_terms = False
        if c.is_number and c < 0:
            swap_terms = True
        elif isinstance(c, sy.Mul):
            cc = c.as_two_terms()[0]
            if cc.is_number and cc < 0:
                swap_terms = True
        elif isinstance(d, sy.Mul):
            dd = d.as_two_terms()[0]
            if dd.is_number and dd < 0:
                swap_terms = True

        if swap_terms:
            result = (-d + (-c)) * b

    # Default to recursed result if no transformation applied
    if result is None:
        result = a * eat_minus_1(b, check=False)

    if check:
        # Verify transformation preserves mathematical meaning
        expanded_orig = expr.expand()
        expanded_result = result.expand()
        if expanded_orig == -1 * expanded_result:
            # Correction for over-optimistic assumption
            result = expr

        assert expanded_orig == expanded_result, f"{expr} != {result} must be identical"

    return result
