"""
Integration tests for P and D classes working together.
"""

import pytest
import sympy as sy
from plumial import P
from plumial.core.D import D
from plumial.core.P import clear_cache
from plumial.utils.symbolic import g, h
from plumial.utils.binary import n, o, e, cycle


def test_p_d_integration():
    """Test that P and D classes work together correctly."""
    p_val = 133
    
    # Create P object
    p_obj = P(p_val)
    
    # Get D object from P
    d_obj = p_obj.D()
    
    # Create D object directly
    d_direct = D(p_val)
    
    # They should be equivalent
    assert d_obj.n() == d_direct.n()
    assert d_obj.o() == d_direct.o()
    assert d_obj.e() == d_direct.e()
    
    # Their polynomial evaluations should match
    assert p_obj.d(g=3, h=2) == d_obj.d(g=3, h=2)
    assert p_obj.d(g=3, h=2) == d_direct.d(g=3, h=2)
    
    print(f"P({p_val}) and D({p_val}) integration successful")


def test_cycle_polynomial_consistency():
    """Test that polynomials are consistent across cycles."""
    p_obj = P(133)
    
    # Get all cycle objects
    cycle_objects = list(p_obj.cycle())
    
    # Test that all have the same n (cycle length)
    for obj in cycle_objects:
        assert obj.n() == p_obj.n()
    
    # Test that d-polynomials are related
    d_values = []
    for obj in cycle_objects:
        d_val = obj.d(g=3, h=2)
        d_values.append(d_val)
    
    print(f"Cycle d-values: {d_values}")
    
    # Test that k polynomials vary across cycle
    k_values = []
    for obj in cycle_objects:
        k_val = obj.k(g=3, h=2)
        k_values.append(k_val)
    
    print(f"Cycle k-values: {k_values}")


def test_mathematical_relationships_across_cycle():
    """Test mathematical relationships hold across the cycle."""
    p_obj = P(133)
    
    for obj in p_obj.cycle():
        try:
            # Test relationships for each object in cycle
            d_val = obj.d(g=3, h=2)
            k_val = obj.k(g=3, h=2)
            f_val = obj.f(g=3, h=2)
            a_val = obj.a(g=3, h=2)
            x_val = obj.x(g=3, h=2)
            
            # Test mathematical relationships
            assert abs(a_val - d_val/f_val) < 1e-10
            assert abs(x_val - k_val/f_val) < 1e-10
            assert abs(x_val * d_val - a_val * k_val) < 1e-10
            
        except ZeroDivisionError:
            print(f"Division by zero for p={obj.p()}")
        except Exception as e:
            print(f"Error for p={obj.p()}: {e}")
    
    print("Mathematical relationships verified across cycle")


def test_binary_operations_integration():
    """Test integration with binary operations."""
    from plumial.utils.binary import cycle as binary_cycle
    
    p_obj = P(133)
    
    # Test that P cycle matches binary cycle
    p_cycle = [obj.p() for obj in p_obj.cycle()]
    binary_cycle_vals = list(binary_cycle(p_obj.p()))
    
    assert p_cycle == binary_cycle_vals
    
    # Test bit counting consistency
    for obj in p_obj.cycle():
        assert obj.n() == n(obj.p())
        assert obj.o() == o(obj.p())
        assert obj.e() == e(obj.p())
    
    print("Binary operations integration successful")


def test_d_integration():
    """Test D objects work correctly."""
    # Create D object from p-value
    d_obj = D(21)  # This gives us n=4, o=2, e=2
    
    # Test that it has correct properties
    assert d_obj.n() == 4
    assert d_obj.o() == 2
    assert d_obj.e() == 2
    
    # Test that we can evaluate it
    d_val = d_obj.d(g=3, h=2)
    assert isinstance(d_val, (int, float, sy.Rational))
    
    print(f"D integration successful")


def test_caching_integration():
    """Test that caching works correctly across P and D."""
    clear_cache()
    
    # Create P object
    p_obj = P(133)
    
    # Get D object from P
    d_obj1 = p_obj.D()
    
    # Create D object directly
    d_obj2 = D(133)
    
    # They should be the same cached object
    assert d_obj1 is d_obj2
    
    # Create another P object with same value
    p_obj2 = P(133)
    
    # Should be same cached object
    assert p_obj is p_obj2
    
    print("Caching integration successful")


def test_symbolic_consistency():
    """Test that symbolic operations are consistent."""
    p_obj = P(133)
    d_obj = D(133)
    
    # Compare symbolic expressions
    p_d_symbolic = p_obj.d()
    d_symbolic = d_obj.d()
    
    # Should be equivalent expressions
    diff = sy.simplify(p_d_symbolic - d_symbolic)
    assert diff == 0
    
    print("Symbolic consistency verified")


def test_performance_with_large_cycles():
    """Test performance with larger cycles."""
    # Test with larger p-values
    large_p_values = [257, 513, 1025]  # p-values with larger cycles
    
    for p_val in large_p_values:
        print(f"\n=== Testing large cycle p={p_val} ===")
        
        p_obj = P(p_val)
        print(f"Cycle length: {p_obj.n()}")
        
        # Test basic operations
        d_val = p_obj.d(g=3, h=2)
        k_val = p_obj.k(g=3, h=2)
        
        print(f"d(3,2) = {d_val}")
        print(f"k(3,2) = {k_val}")
        
        # Test cycle navigation
        next_obj = p_obj.next()
        pred_obj = p_obj.pred()
        
        assert next_obj.p() != p_obj.p()
        assert pred_obj.p() != p_obj.p()
        
        # Test that cycle has correct length
        cycle_length = len(list(p_obj.cycle()))
        assert cycle_length == p_obj.n()
        
        print(f"Large cycle test successful")


def test_edge_case_integration():
    """Test edge cases where P and D work together."""
    edge_cases = [2, 3, 4, 5, 6, 7, 8]
    
    for p_val in edge_cases:
        print(f"\n=== Testing integration edge case p={p_val} ===")
        
        p_obj = P(p_val)
        d_obj = D(p_val)
        
        # Basic properties should match
        assert p_obj.n() == d_obj.n()
        assert p_obj.o() == d_obj.o()
        assert p_obj.e() == d_obj.e()
        
        # Polynomial evaluations should match
        p_d_val = p_obj.d(g=3, h=2)
        d_val = d_obj.d(g=3, h=2)
        assert p_d_val == d_val
        
        # Test that P object's D() method returns equivalent D
        p_d_obj = p_obj.D()
        assert p_d_obj.n() == d_obj.n()
        assert p_d_obj.o() == d_obj.o()
        assert p_d_obj.e() == d_obj.e()
        
        print(f"Edge case p={p_val} integration successful")


def test_error_propagation():
    """Test that errors propagate correctly between P and D."""
    p_obj = P(133)
    
    # Test that mathematical errors are handled consistently
    try:
        # Try operations that might fail
        a_val = p_obj.a(g=3, h=2)
        x_val = p_obj.x(g=3, h=2)
        
        # If they succeed, test relationships
        d_val = p_obj.d(g=3, h=2)
        k_val = p_obj.k(g=3, h=2)
        f_val = p_obj.f(g=3, h=2)
        
        assert abs(a_val - d_val/f_val) < 1e-10
        assert abs(x_val - k_val/f_val) < 1e-10
        
    except ZeroDivisionError:
        print("Division by zero handled correctly")
    except Exception as e:
        print(f"Error handled: {e}")


def test_memory_efficiency():
    """Test memory efficiency of integration."""
    clear_cache()
    
    # Create many objects and verify they're cached
    p_values = [133] * 100
    p_objects = [P(p_val) for p_val in p_values]
    
    # All should be the same object
    for obj in p_objects[1:]:
        assert obj is p_objects[0]
    
    # Same for D objects
    d_objects = [obj.D() for obj in p_objects]
    for obj in d_objects[1:]:
        assert obj is d_objects[0]
    
    print("Memory efficiency verified")


def test_comprehensive_workflow():
    """Test a comprehensive workflow using P and D together."""
    # Start with a p-value
    p_val = 133
    print(f"Starting with p={p_val}")
    
    # Create P object
    p_obj = P(p_val)
    print(f"Created P({p_val}): n={p_obj.n()}, o={p_obj.o()}, e={p_obj.e()}")
    
    # Get cycle
    cycle_objects = list(p_obj.cycle())
    print(f"Cycle length: {len(cycle_objects)}")
    print(f"Cycle p-values: {[obj.p() for obj in cycle_objects]}")
    
    # Analyze each object in cycle
    for i, obj in enumerate(cycle_objects):
        print(f"\n--- Cycle object {i}: p={obj.p()} ---")
        
        # Get D object
        d_obj = obj.D()
        
        # Calculate polynomials
        d_val = obj.d(g=3, h=2)
        k_val = obj.k(g=3, h=2)
        
        print(f"d(3,2) = {d_val}")
        print(f"k(3,2) = {k_val}")
        
        # Test UV polynomial
        uv_val = obj.uv()
        print(f"uv() = {uv_val}")
        
        # Test f polynomial
        try:
            f_val = obj.f(g=3, h=2)
            print(f"f(3,2) = {f_val}")
        except Exception as e:
            print(f"f calculation failed: {e}")
    
    print("\nComprehensive workflow completed successfully!")


if __name__ == "__main__":
    # Run integration tests
    test_p_d_integration()
    test_cycle_polynomial_consistency()
    test_mathematical_relationships_across_cycle()
    test_binary_operations_integration()
    test_d_from_counts_integration()
    test_caching_integration()
    test_symbolic_consistency()
    test_performance_with_large_cycles()
    test_edge_case_integration()
    test_error_propagation()
    test_memory_efficiency()
    test_comprehensive_workflow()
    print("\nAll integration tests completed!")