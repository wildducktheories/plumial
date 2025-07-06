"""
Tests for P instance caching functionality.
"""

import pytest
from plumial.core.polynomials import P, clear_cache, cache_info


def test_p_caching():
    """Test that P instances are properly cached."""
    clear_cache()
    
    # Create two instances with same p-value
    p1 = P(133)
    p2 = P(133)
    
    # Should be the same object (cached)
    assert p1 is p2
    
    # Cache should have one hit
    info = cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_p_caching_with_relationships():
    """Test that P instances with pred/next are not cached."""
    clear_cache()
    
    p1 = P(133)
    p2 = P(133)
    
    # These should be cached and identical
    assert p1 is p2
    
    # Create with pred/next - should not use cache
    p3 = P(133, pred=p1)
    p4 = P(133, next=p2)
    
    # These should be different instances
    assert p3 is not p1
    assert p4 is not p2
    assert p3 is not p4


def test_p_basic_properties():
    """Test basic P instance properties."""
    p = P(133)  # Binary: 10000101
    
    assert p.p() == 133
    assert p.n() == 7  # bit_length() - 1 = 8 - 1 = 7
    assert p.o() == 2  # bit_count() - 1 = 3 - 1 = 2
    assert p.e() == 5  # n - o = 7 - 2 = 5


def test_p_cycle_navigation():
    """Test cycle navigation methods."""
    p = P(133)
    
    # Test next/pred relationship
    next_p = p.next()
    assert next_p.pred().p() == p.p()
    
    # Test cycle length matches n
    cycle_elements = list(p.cycle())
    assert len(cycle_elements) == p.n()


def test_cache_clearing():
    """Test cache clearing functionality."""
    clear_cache()
    
    p1 = P(133)
    info1 = cache_info()
    assert info1.currsize == 1
    
    clear_cache()
    info2 = cache_info()
    assert info2.currsize == 0
    
    # New instance should be different after cache clear
    p2 = P(133)
    assert p1 is not p2  # Different instances after cache clear


def test_p_string_representation():
    """Test string representations."""
    p = P(133)
    
    assert str(p) == "133"
    assert repr(p) == "P(133)"


def test_p_equality_and_hashing():
    """Test equality and hashing."""
    p1 = P(133)
    p2 = P(133)
    p3 = P(134)
    
    assert p1 == p2
    assert p1 != p3
    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)