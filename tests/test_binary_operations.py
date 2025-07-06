"""
Tests for binary operations module.
"""

import pytest

from plumial import P
from plumial.utils.binary import (
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


def test_bit_counting_functions():
    """Test n(), o(), e() functions."""
    # Test with p=133 (binary: 10000101)
    p_val = 133

    assert n(p_val) == 7  # bit_length - 1
    assert o(p_val) == 2  # bit_count - 1 (3 ones - 1 = 2)
    assert e(p_val) == 5  # n - o = 7 - 2 = 5

    # Test with p=9 (binary: 1001)
    p_val = 9
    assert n(p_val) == 3  # bit_length - 1
    assert o(p_val) == 1  # bit_count - 1 (2 ones - 1 = 1)
    assert e(p_val) == 2  # n - o = 3 - 1 = 2


def test_is_odd():
    """Test is_odd() function."""
    assert is_odd(9) == True
    assert is_odd(10) == False
    assert is_odd(133) == True
    assert is_odd(134) == False


def test_next_p_and_pred_p():
    """Test next_p() and pred_p() functions."""
    # Test that pred_p(next_p(p)) == p
    test_values = [9, 133, 17, 65]

    for p_val in test_values:
        next_val = next_p(p_val)
        pred_val = pred_p(next_val)
        assert (
            pred_val == p_val
        ), f"pred_p(next_p({p_val})) = {pred_val}, expected {p_val}"

        # Test reverse
        pred_val = pred_p(p_val)
        next_val = next_p(pred_val)
        assert (
            next_val == p_val
        ), f"next_p(pred_p({p_val})) = {next_val}, expected {p_val}"


def test_cycle_generation():
    """Test cycle() function."""
    # Test with p=9
    p_val = 9
    cycle_values = list(cycle(p_val))

    # Cycle should have length n(p)
    assert len(cycle_values) == n(p_val)

    # All values should be unique
    assert len(set(cycle_values)) == len(cycle_values)

    # First value should be p_val
    assert cycle_values[0] == p_val

    # Each next value should match next_p()
    for i in range(len(cycle_values) - 1):
        expected_next = next_p(cycle_values[i])
        assert cycle_values[i + 1] == expected_next


def test_binary_string_conversion():
    """Test binary_string() and from_binary_string() functions."""
    test_values = [9, 133, 17, 65]

    for val in test_values:
        # Convert to binary string and back
        binary_str = binary_string(val)
        converted_back = from_binary_string(binary_str)
        assert converted_back == val

        # Test with width padding
        binary_str_padded = binary_string(val, width=10)
        assert len(binary_str_padded) == 10
        assert binary_str_padded.endswith(binary_str)

        converted_back_padded = from_binary_string(binary_str_padded)
        assert converted_back_padded == val


def test_odd_bit_indices():
    """Test odd_bit_indices() function."""

    # Create mock P object for testing
    class MockP:
        def __init__(self, p_val):
            self._p = p_val

        def p(self):
            return self._p

        def n(self):
            return self._p.bit_length() - 1

    # Test with p=133 (binary: 10000101)
    # Only cycle bits matter: lower 7 bits are 0000101
    # So odd bits are at positions 0 and 2 (not 7, which is the length bit)
    mock_p = MockP(133)

    # Vector form
    vector_indices = list(odd_bit_indices(mock_p, matrix=False))
    # Should have indices where cycle bits are 1 (positions 0 and 2)
    assert len(vector_indices) == 2
    assert (0,) in vector_indices
    assert (2,) in vector_indices

    # Matrix form
    matrix_indices = list(odd_bit_indices(mock_p, matrix=True))
    assert len(matrix_indices) == 2
    # Should be (o_index, e_index) pairs


def test_bit_pattern_analysis():
    """Test bit_pattern_analysis() function."""
    p_val = 133
    analysis = bit_pattern_analysis(p_val)

    # Check required fields
    assert "binary" in analysis
    assert "n" in analysis
    assert "o" in analysis
    assert "e" in analysis
    assert "odd_positions" in analysis
    assert "even_positions" in analysis
    assert "cycle_length" in analysis

    # Check values
    assert analysis["binary"] == binary_string(p_val)
    assert analysis["n"] == n(p_val)
    assert analysis["o"] == o(p_val)
    assert analysis["e"] == e(p_val)
    assert analysis["cycle_length"] == n(p_val)

    # Check positions
    assert len(analysis["odd_positions"]) == o(p_val)
    assert len(analysis["even_positions"]) == e(p_val)

    print(f"Bit pattern analysis for p={p_val}: {analysis}")


def test_validate_p_value():
    """Test validate_p_value() function."""
    # Valid p-values
    assert validate_p_value(1) == True
    assert validate_p_value(9) == True
    assert validate_p_value(133) == True

    # Invalid p-values
    assert validate_p_value(0) == False
    assert validate_p_value(-1) == False
    assert validate_p_value(0.5) == False
    assert validate_p_value("9") == False


def test_p_from_binary_pattern():
    """Test p_from_binary_pattern() function."""
    # Test with known patterns
    pattern = "0000101"
    p_val = p_from_binary_pattern(pattern)
    expected = from_binary_string("1" + pattern)  # Add MSB
    assert p_val == expected

    # Test empty pattern
    p_val = p_from_binary_pattern("")
    assert p_val == 1

    # Test that result is valid p-value
    patterns = ["01", "101", "0101", "1010"]
    for pattern in patterns:
        p_val = p_from_binary_pattern(pattern)
        assert validate_p_value(p_val)


def test_cycle_statistics():
    """Test cycle_statistics() function."""
    p_val = 133
    stats = cycle_statistics(p_val)

    # Check required fields
    assert "cycle_length" in stats
    assert "min_p" in stats
    assert "max_p" in stats
    assert "odd_count" in stats
    assert "even_count" in stats
    assert "cycle_values" in stats

    # Check values
    assert stats["cycle_length"] == n(p_val)
    assert len(stats["cycle_values"]) == n(p_val)
    assert stats["min_p"] == min(stats["cycle_values"])
    assert stats["max_p"] == max(stats["cycle_values"])

    # Check counts
    expected_odd_count = sum(o(pval) for pval in stats["cycle_values"])
    expected_even_count = sum(e(pval) for pval in stats["cycle_values"])
    assert stats["odd_count"] == expected_odd_count
    assert stats["even_count"] == expected_even_count

    print(f"Cycle statistics for p={p_val}: {stats}")


def test_compatibility_with_original():
    """Test compatibility with original bits.py functions."""
    # Import original functions if available
    try:
        import os
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../collatz-private/src")
        )

        from collatz import bits as orig_bits

        # Test values
        test_values = [9, 133, 17, 65]

        for p_val in test_values:
            # Compare bit counting functions
            assert n(p_val) == orig_bits.n(p_val), f"n({p_val}) mismatch"
            assert o(p_val) == orig_bits.o(p_val), f"o({p_val}) mismatch"
            assert e(p_val) == orig_bits.e(p_val), f"e({p_val}) mismatch"

            # Compare cycle navigation
            assert next_p(p_val) == orig_bits.next(p_val), f"next_p({p_val}) mismatch"
            assert pred_p(p_val) == orig_bits.pred(p_val), f"pred_p({p_val}) mismatch"

            # Compare binary string conversion
            assert binary_string(p_val) == orig_bits.binary(
                p_val
            ), f"binary_string({p_val}) mismatch"

            # Compare cycle generation
            our_cycle = list(cycle(p_val))
            orig_cycle = list(orig_bits.cycle(p_val))
            assert our_cycle == orig_cycle, f"cycle({p_val}) mismatch"

        print("All compatibility tests passed!")

    except ImportError:
        print("Original bits.py not available, skipping compatibility tests")


def test_edge_cases():
    """Test edge cases for binary operations."""
    # Test with minimal p-values (skip p=1 since it has n=0)
    edge_cases = [2, 3, 4, 5]

    for p_val in edge_cases:
        print(f"\n=== Testing edge case p={p_val} ===")

        # Basic functions should work
        assert n(p_val) >= 0
        assert o(p_val) >= 0
        assert e(p_val) >= 0

        # Cycle operations
        cycle_vals = list(cycle(p_val))
        assert len(cycle_vals) == n(p_val)

        # Navigation operations (only if n > 0)
        if n(p_val) > 0:
            next_val = next_p(p_val)
            pred_val = pred_p(p_val)
            assert pred_p(next_val) == p_val

        # Binary operations
        binary_str = binary_string(p_val)
        assert from_binary_string(binary_str) == p_val

        # Validation
        assert validate_p_value(p_val) == True

        print(f"Edge case p={p_val} passed all tests")

    # Special case: p=1 (n=0, no cycle bits)
    print(f"\n=== Testing special case p=1 ===")
    p_val = 1
    assert n(p_val) == 0
    assert o(p_val) == 0  # bit_count - 1 = 1 - 1 = 0
    assert e(p_val) == 0  # n - o = 0 - 0 = 0
    assert validate_p_value(p_val) == True
    print("Special case p=1 passed basic tests")


def test_integration_with_p_class():
    """Test integration with P class."""
    # Test that binary operations work with P class
    p_val = 133
    p_obj = P(p_val)

    # Binary operations should work with p_obj.p()
    assert n(p_obj.p()) == p_obj.n()
    assert o(p_obj.p()) == p_obj.o()
    assert e(p_obj.p()) == p_obj.e()

    # Cycle navigation should be consistent
    next_val = next_p(p_obj.p())
    next_obj = p_obj.next()
    assert next_val == next_obj.p()

    pred_val = pred_p(p_obj.p())
    pred_obj = p_obj.pred()
    assert pred_val == pred_obj.p()

    print("Integration with P class successful!")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_bit_counting_functions()
    test_is_odd()
    test_next_p_and_pred_p()
    test_cycle_generation()
    test_binary_string_conversion()
    test_odd_bit_indices()
    test_bit_pattern_analysis()
    test_validate_p_value()
    test_p_from_binary_pattern()
    test_cycle_statistics()
    test_compatibility_with_original()
    test_edge_cases()
    test_integration_with_p_class()
    print("\nAll binary operations tests completed!")
