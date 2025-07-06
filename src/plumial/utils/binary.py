"""
Binary operations for Collatz conjecture analysis.

This module provides comprehensive bit manipulation functions for working with p-values,
cycle navigation, and binary representations used in Collatz sequence analysis.

The module implements the core bit-level operations that underlie the mathematical
structure of Collatz sequences, including:

Key Concepts:
    - P-values: Binary-encoded representations of Collatz sequence paths
    - MSB encoding: The most significant bit encodes the path length
    - Cycle bits: Lower bits encode the actual sequence operations
    - Odd/Even bits: Correspond to 3x+1 and halving operations respectively

Functions:
    - Bit counting: n(), o(), e() for analyzing bit structure
    - Cycle navigation: next_p(), pred_p(), cycle() for traversing sequences
    - Binary conversion: binary_string(), from_binary_string() for representations
    - Analysis: bit_pattern_analysis(), cycle_statistics() for detailed examination

Mathematical Background:
    P-values use a special encoding where:
    - MSB (position n): Length bit (always 1)
    - Lower n bits: Actual cycle operations
    - Odd bits (1): Correspond to 3x+1 operations
    - Even bits (0): Correspond to halving operations

Examples:
    >>> p = 133  # Binary: 10000101
    >>> n(p)  # 7 (total path bits)
    >>> o(p)  # 2 (odd operations: 3x+1)
    >>> e(p)  # 5 (even operations: halving)
    >>> list(cycle(p))  # Full cycle starting from p=133
"""

import math
from typing import Iterator, Tuple, Union, Optional, Dict, List, Any

# Import type aliases for better type safety  
from ..types import (
    BitPosition, BitPositionIterator, CycleIterator, BinaryPattern,
    CycleStatistics, BitAnalysis, PValueProtocol
)


def n(p: int) -> int:
    """
    Calculate the number of path bits implied by p.
    
    Args:
        p: The p-value (must be > 0)
        
    Returns:
        Number of path bits (bit_length - 1)
        
    Examples:
        >>> n(133)  # 133 = 0b10000101
        7
    """
    return p.bit_length() - 1


def o(p: int) -> int:
    """
    Calculate the number of 'odd' bits in p.
    
    Args:
        p: The p-value (must be > 0)
        
    Returns:
        Number of odd bits (bit_count - 1)
        
    Examples:
        >>> o(133)  # 133 = 0b10000101 has 3 ones
        2
    """
    return p.bit_count() - 1


def e(p: int) -> int:
    """
    Calculate the number of 'even' bits in p.
    
    Args:
        p: The p-value (must be > 0)
        
    Returns:
        Number of even bits (n - o)
        
    Examples:
        >>> e(133)  # n=7, o=2, so e=5
        5
    """
    return n(p) - o(p)


def is_odd(p: int) -> bool:
    """
    Check if p represents an odd step in Collatz sequence.
    
    Args:
        p: The p-value
        
    Returns:
        True if p is odd, False otherwise
    """
    return p % 2 == 1


def next_p(p: int) -> int:
    """
    Rotate the MSB-1 into the LSB and shift other bits left.
    
    This implements the bit rotation for cycle navigation in Collatz analysis.
    
    Args:
        p: The p-value
        
    Returns:
        Next p-value in the cycle
        
    Examples:
        >>> next_p(133)  # Binary rotation
        198
    """
    _n = n(p)
    return ((p - (1 << _n)) | (p & 1) << _n) >> 1 | (1 << _n)


def pred_p(p: int) -> int:
    """
    Rotate the lower n-1 bits of p to the left (predecessor operation).
    
    Args:
        p: The p-value
        
    Returns:
        Previous p-value in the cycle
        
    Examples:
        >>> pred_p(next_p(133))
        133
    """
    _n = n(p)
    if _n == 0:
        return p  # Special case: no cycle bits to rotate
    
    stop = 1 << _n  # MSB used to encode the length
    mask = stop - 1  # every bit lower than the stop bit
    
    # shift the lower bits left, the high bit right, add back the stop bit
    return (p << 1) & mask | (p & mask) >> (_n - 1) | stop


def cycle(p: int, first_only: bool = False) -> Iterator[int]:
    """
    Enumerate the cycle of p values.
    
    Args:
        p: Starting p-value
        first_only: If True, stop after the first repetition
        
    Yields:
        p-values in the cycle
        
    Examples:
        >>> list(cycle(9))  # Full cycle
        [9, 10, 5, 6, 3, 8, 4, 2]
    """
    c = 0
    _n = n(p)
    
    if first_only:
        s = set()
        
    while c < _n or (first_only and p not in s):
        c += 1
        if first_only:
            s.add(p)
        yield p
        p = next_p(p)


def binary_string(b: int, width: int = 0) -> BinaryPattern:
    """
    Generate binary string representation of integer.
    
    Args:
        b: Integer to convert
        width: Minimum width (right-justified with zeros)
        
    Returns:
        Binary string representation
        
    Examples:
        >>> binary_string(133)
        '10000101'
        >>> binary_string(133, 10)
        '0010000101'
    """
    return format(b, 'b').rjust(width, '0')


def from_binary_string(s: BinaryPattern) -> int:
    """
    Convert binary string to integer.
    
    Args:
        s: Binary string (e.g., '10000101')
        
    Returns:
        Integer value
        
    Examples:
        >>> from_binary_string('10000101')
        133
    """
    return int(s, 2)


def odd_bit_indices(p_obj: PValueProtocol, matrix: bool = False) -> BitPositionIterator:
    """
    Generate indices of odd bits in a p-value.
    
    Args:
        p_obj: P object with p() and n() methods
        matrix: If True, return matrix indices (o,e), else vector indices (i,)
        
    Yields:
        Tuples of indices where odd bits occur
        
    Examples:
        >>> p = P(133)
        >>> list(odd_bit_indices(p))  # Vector form
        [(0,), (7,)]
        >>> list(odd_bit_indices(p, matrix=True))  # Matrix form  
        [(0, 0), (1, 5)]
    """
    w = p_obj.p()
    
    if not matrix:
        # Vector form: yield (i,) for each odd bit position
        for i in range(p_obj.n()):
            if w % 2 == 1:
                yield (i,)
            w >>= 1
    else:
        # Matrix form: yield (o_index, e_index) for each odd bit
        e_count = 0
        o_count = 0
        for i in range(p_obj.n()):
            if w % 2 == 1:
                yield (o_count, e_count)
                o_count += 1
            else:
                e_count += 1
            w >>= 1


def bit_pattern_analysis(p: int) -> BitAnalysis:
    """
    Analyze the bit pattern of a p-value.
    
    Args:
        p: The p-value
        
    Returns:
        Dictionary with pattern analysis
        
    Examples:
        >>> bit_pattern_analysis(133)
        {
            'binary': '10000101',
            'n': 7, 'o': 2, 'e': 5,
            'odd_positions': [0, 7],
            'even_positions': [1, 2, 3, 4, 5, 6]
        }
    """
    binary_repr = binary_string(p)
    _n = n(p)
    _o = o(p)
    _e = e(p)
    
    # Find positions of odd and even bits (from right, 0-indexed)
    odd_positions = []
    even_positions = []
    
    for i in range(_n):
        if (p >> i) & 1:
            odd_positions.append(i)
        else:
            even_positions.append(i)
    
    return {
        'binary': binary_repr,
        'n': _n,
        'o': _o, 
        'e': _e,
        'odd_positions': odd_positions,
        'even_positions': even_positions,
        'cycle_length': _n
    }


def validate_p_value(p: int) -> bool:
    """
    Validate that p is a proper p-value.
    
    Args:
        p: Value to validate
        
    Returns:
        True if p is a valid p-value, False otherwise
        
    Examples:
        >>> validate_p_value(133)
        True
        >>> validate_p_value(0)
        False
    """
    if not isinstance(p, int) or p <= 0:
        return False
    
    # Check that the MSB is set (required for p-value encoding)
    if p.bit_length() == 0:
        return False
        
    # MSB should be 1 for proper p-value encoding
    msb_position = p.bit_length() - 1
    return (p >> msb_position) & 1 == 1


def p_from_binary_pattern(pattern: str) -> int:
    """
    Create p-value from binary pattern string.
    
    Args:
        pattern: Binary pattern (without leading MSB)
        
    Returns:
        p-value with MSB set
        
    Examples:
        >>> p_from_binary_pattern('0000101')
        133  # '1' + '0000101' = '10000101'
    """
    if not pattern:
        return 1  # Minimal p-value
    
    # Add MSB and convert
    full_pattern = '1' + pattern
    return from_binary_string(full_pattern)


def cycle_statistics(p: int) -> CycleStatistics:
    """
    Calculate statistics for a p-value cycle.
    
    Args:
        p: Starting p-value
        
    Returns:
        Dictionary with cycle statistics
    """
    cycle_values = list(cycle(p))
    
    return {
        'cycle_length': len(cycle_values),
        'min_p': min(cycle_values),
        'max_p': max(cycle_values),
        'odd_count': sum(o(pval) for pval in cycle_values),
        'even_count': sum(e(pval) for pval in cycle_values),
        'cycle_values': cycle_values
    }