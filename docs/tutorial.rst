Tutorial
========

This tutorial introduces the key concepts and usage patterns of the Plumial library for Collatz conjecture analysis.

Getting Started
---------------

The Plumial library provides tools for analyzing the Collatz conjecture through polynomial representations. The core idea is to encode the binary path information of Collatz sequences as polynomials.

Basic Usage
-----------

Creating P Objects
~~~~~~~~~~~~~~~~~~

The fundamental object in Plumial is the P class, which represents a path polynomial:

.. code-block:: python

   from plumial import P
   
   # Create a polynomial for p-value 133
   p = P(133)
   
   # Get basic properties
   print(f"p-value: {p.p()}")  # 133
   print(f"Total bits: {p.n()}")  # 7
   print(f"Odd bits: {p.o()}")  # 2  
   print(f"Even bits: {p.e()}")  # 5

Working with Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~

P objects provide several polynomial representations:

.. code-block:: python

   # d-polynomial (symbolic)
   print(p.d())  # h**5 - g**2
   
   # Evaluate numerically
   print(p.d(g=3, h=2))  # 23
   
   # K polynomial
   print(p.k())  # Symbolic k polynomial
   print(p.k(g=3, h=2))  # Numerical evaluation
   
   # UV polynomial representation
   print(p.uv())  # Polynomial in u and v variables

Cycle Navigation
~~~~~~~~~~~~~~~~

P objects can navigate through their cycles:

.. code-block:: python

   # Get next/previous elements
   next_p = p.next()
   prev_p = p.pred()
   
   # Iterate through the entire cycle
   for p_obj in p.cycle():
       print(p_obj.p())

Symbolic Mathematics
--------------------

Plumial provides three namespace objects for mathematical work:

.. code-block:: python

   from plumial.utils import S, I, F
   
   # S: Symbol namespace
   print(S.g, S.h, S.u, S.v)  # Basic symbols
   
   # I: Indexed symbols  
   print(I.p[0], I.x[S.i])  # Indexed symbols
   
   # F: Functions namespace
   print(F.isodd(p))  # True if p-value is odd
   print(F.n(p))  # Get bit count

Functional Operations
---------------------

The F namespace provides functional programming utilities:

.. code-block:: python

   # Create curried functions
   k_func = F.k(g=3, h=2)  # Returns lambda p: p.k(3, 2)
   
   # Use with cycle operations
   k_values = list(p.cycle(map=k_func))
   
   # Filter odd elements and map to k values
   odd_k_values = list(p.cycle(map=F.k(), filter=F.isodd))
   
   # Compose operations
   composed = F.compose(F.n, F.next)
   result = composed(p)  # n(next(p))

Data Analysis
-------------

Convert cycle data to pandas DataFrames for analysis:

.. code-block:: python

   import pandas as pd
   from plumial.utils import F
   
   # Convert cycle to DataFrame
   df = F.to_dataframe(P(133).cycle(), g=3, h=2)
   print(df.head())
   
   # Analyze patterns
   print(f"Average k value: {df['k'].mean()}")
   print(f"Odd elements: {df[df['p'] % 2 == 1].shape[0]}")

Advanced Topics
---------------

D Polynomials
~~~~~~~~~~~~~

Work directly with D objects for algebraic analysis:

.. code-block:: python

   from plumial import D
   
   # Create d polynomial from p-value
   d = D(133)
   
   # Mathematical operations
   print(d.d())  # h**5 - g**2
   print(d.gcd_result())  # GCD analysis

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Plumial uses LRU caching for performance:

.. code-block:: python

   from plumial.core.polynomials import cache_info, clear_cache
   
   # Check cache statistics
   info = cache_info()
   print(f"Cache hits: {info.hits}, misses: {info.misses}")
   
   # Clear cache if needed
   clear_cache()

Type Safety
~~~~~~~~~~~

Plumial provides comprehensive type hints:

.. code-block:: python

   from plumial.types import PValueProtocol, OptionalNumeric
   from typing import Iterator
   
   def process_cycle(p: PValueProtocol) -> Iterator[int]:
       """Type-safe cycle processing."""
       for obj in p.cycle():
           yield obj.p()

Mathematical Background
-----------------------

The Collatz conjecture involves sequences defined by:

- If n is even: n → n/2
- If n is odd: n → 3n + 1

Plumial encodes the binary path of these operations as polynomials:

- Path objects (P): Hydrated path identifiers that encode the sequence of operations
- D polynomials (D): h^e - g^o where e=even bits, o=odd bits  
- UV polynomials: Binary path representation using u and v variables
- K polynomials: Derived from UV→GH transformations

This mathematical framework enables algebraic analysis of Collatz behavior and cycle detection.