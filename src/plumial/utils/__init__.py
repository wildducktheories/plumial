"""
Utility modules for Plumial package.
"""

from .binary import (
    binary_string,
    bit_pattern_analysis,
    cycle,
    cycle_statistics,
    e,
    from_binary_string,
    is_odd,
    n,
    next_p,
    o,
    odd_bit_indices,
    p_from_binary_pattern,
    pred_p,
    validate_p_value,
)
from .matrix_utils import eat_minus_1, expr_as_vector, vector_as_expr
from .symbolic import (
    TRANSFORM_GH_TO_UV,
    TRANSFORM_UV_TO_GH,
    F,
    FunctionsNamespace,
    I,
    IndexedNamespace,
    S,
    SymbolNamespace,
    a,
    additional_symbols,
    all_symbols,
    b,
    b_indexed,
    bit_count_symbols,
    c,
    constants,
    create_polynomial_expression,
    d,
)
from .symbolic import e as e_sym
from .symbolic import (
    e_indexed,
    extract_coefficients,
    f,
    g,
    get_indexed_symbol,
    get_symbol,
    gh_to_uv_transform,
    h,
    i,
    index_symbols,
    indexed_dict,
    indexed_symbols,
    j,
    k,
    k_indexed,
)
from .symbolic import (
    m as m_sym,  # Primary symbols; Bit count and index symbols (renamed to avoid conflicts with binary functions); Constants; Substitution dictionaries; Transformation constants; Indexed symbols; Collections; Utility functions; Compatibility; Lookup dictionaries
)
from .symbolic import m_indexed
from .symbolic import n as n_sym
from .symbolic import o as o_sym
from .symbolic import o_indexed
from .symbolic import p as p_sym
from .symbolic import (
    p_indexed,
    polynomial_symbols,
    primary_symbols,
    q,
    r,
    substitute_values,
    substitution_dict,
    symbol_dict,
    three,
    two,
    u,
    uv_symbols,
    uv_to_gh_transform,
    v,
    x,
    x_indexed,
)

__all__ = [
    # Primary symbols
    "g",
    "h",
    "u",
    "v",
    "a",
    "d",
    "k",
    "x",
    "f",
    # Bit count symbols (renamed to avoid conflicts)
    "m_sym",
    "n_sym",
    "o_sym",
    "e_sym",
    "i",
    "j",
    "p_sym",
    "b",
    "c",
    "r",
    "q",
    # Constants
    "two",
    "three",
    # Transformation constants
    "TRANSFORM_UV_TO_GH",
    "TRANSFORM_GH_TO_UV",
    # Indexed symbols
    "p_indexed",
    "x_indexed",
    "k_indexed",
    "b_indexed",
    "e_indexed",
    "o_indexed",
    "m_indexed",
    # Symbol collections
    "primary_symbols",
    "polynomial_symbols",
    "uv_symbols",
    "bit_count_symbols",
    "index_symbols",
    "additional_symbols",
    "constants",
    "indexed_symbols",
    "all_symbols",
    # Symbolic utility functions
    "get_symbol",
    "get_indexed_symbol",
    "substitute_values",
    "create_polynomial_expression",
    "uv_to_gh_transform",
    "gh_to_uv_transform",
    "extract_coefficients",
    # Compatibility
    "S",
    "I",
    "F",
    "SymbolNamespace",
    "IndexedNamespace",
    "FunctionsNamespace",
    "symbol_dict",
    "indexed_dict",
    "substitution_dict",
    # Matrix utilities
    "expr_as_vector",
    "vector_as_expr",
    "eat_minus_1",
    # Binary operations (functions)
    "n",
    "o",
    "e",
    "is_odd",
    "next_p",
    "pred_p",
    "cycle",
    "binary_string",
    "from_binary_string",
    "odd_bit_indices",
    "bit_pattern_analysis",
    "validate_p_value",
    "p_from_binary_pattern",
    "cycle_statistics",
]
