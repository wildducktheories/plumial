"""
Utility modules for Plumial package.
"""

from .symbolic import (
    # Primary symbols
    g, h, u, v, a, d, k, x, f,
    # Bit count and index symbols (renamed to avoid conflicts with binary functions)
    m as m_sym, n as n_sym, o as o_sym, e as e_sym, i, j, p as p_sym, b, c, r, q,
    # Constants
    two, three,
    # Substitution dictionaries
    COLLATZ_3_2, COLLATZ_5_2, COLLATZ_7_2, COLLATZ_15_2, COLLATZ_8_3, COLLATZ_5_3, COLLATZ_STD,
    # Transformation constants
    TRANSFORM_UV_TO_GH, TRANSFORM_GH_TO_UV,
    # Indexed symbols
    p_indexed, x_indexed, k_indexed, b_indexed, e_indexed, o_indexed, m_indexed,
    # Collections
    primary_symbols, polynomial_symbols, uv_symbols, bit_count_symbols,
    index_symbols, additional_symbols, constants, indexed_symbols, all_symbols,
    # Utility functions
    get_symbol, get_indexed_symbol, substitute_values, create_polynomial_expression,
    difference_polynomial, uv_to_gh_transform, gh_to_uv_transform, extract_coefficients,
    # Compatibility
    S, I, F, SymbolNamespace, IndexedNamespace, FunctionsNamespace,
    # Lookup dictionaries
    symbol_dict, indexed_dict, substitution_dict
)
from .matrix_utils import expr_as_vector, vector_as_expr, eat_minus_1
from .binary import (
    n, o, e, is_odd, next_p, pred_p, cycle, binary_string, from_binary_string,
    odd_bit_indices, bit_pattern_analysis, validate_p_value, p_from_binary_pattern,
    cycle_statistics
)

__all__ = [
    # Primary symbols
    "g", "h", "u", "v", "a", "d", "k", "x", "f",
    # Bit count symbols (renamed to avoid conflicts)
    "m_sym", "n_sym", "o_sym", "e_sym", "i", "j", "p_sym", "b", "c", "r", "q",
    # Constants
    "two", "three",
    # Substitution dictionaries
    "COLLATZ_3_2", "COLLATZ_5_2", "COLLATZ_7_2", "COLLATZ_15_2", "COLLATZ_8_3", "COLLATZ_5_3", "COLLATZ_STD",
    # Transformation constants
    "TRANSFORM_UV_TO_GH", "TRANSFORM_GH_TO_UV",
    # Indexed symbols
    "p_indexed", "x_indexed", "k_indexed", "b_indexed", "e_indexed", "o_indexed", "m_indexed",
    # Symbol collections
    "primary_symbols", "polynomial_symbols", "uv_symbols", "bit_count_symbols",
    "index_symbols", "additional_symbols", "constants", "indexed_symbols", "all_symbols",
    # Symbolic utility functions
    "get_symbol", "get_indexed_symbol", "substitute_values", "create_polynomial_expression",
    "difference_polynomial", "uv_to_gh_transform", "gh_to_uv_transform", "extract_coefficients",
    # Compatibility
    "S", "I", "F", "SymbolNamespace", "IndexedNamespace", "FunctionsNamespace", "symbol_dict", "indexed_dict", "substitution_dict",
    # Matrix utilities
    "expr_as_vector", "vector_as_expr", "eat_minus_1",
    # Binary operations (functions)
    "n", "o", "e", "is_odd", "next_p", "pred_p", "cycle", "binary_string", 
    "from_binary_string", "odd_bit_indices", "bit_pattern_analysis", 
    "validate_p_value", "p_from_binary_pattern", "cycle_statistics"
]