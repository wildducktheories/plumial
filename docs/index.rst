Plumial: Collatz Conjecture Analysis
=====================================

Plumial is a Python package for mathematical analysis of the Collatz conjecture using polynomial representations. 
It provides efficient tools for working with path objects, d polynomials, and uv transformations.

Features
--------

* **Path Objects (P class)**: Hydrated path identifiers that encode Collatz sequence paths
* **D Polynomials (D class)**: Algebraic analysis using h^e - g^o forms  
* **UV Polynomials**: Binary path representations with u/v variables
* **Symbolic Mathematics**: Full SymPy integration with S, I, F namespaces
* **Performance Optimized**: LRU caching for computational efficiency
* **Type Safe**: Comprehensive type hints and protocols

Quick Start
-----------

.. code-block:: python

   from plumial import P
   from plumial.utils import S, I, F, COLLATZ_STD

   # Create a polynomial for p-value 133
   p = P(133)
   
   # Get bit counts
   print(f"n={p.n()}, o={p.o()}, e={p.e()}")  # n=7, o=2, e=5
   
   # Work with symbolic forms
   print(p.d())  # h**5 - g**2
   print(p.k())  # Symbolic k polynomial
   
   # Evaluate numerically - multiple ways
   print(p.d(g=3, h=2))           # Direct: 23
   print(p.d().subs(COLLATZ_STD)) # Using constants: 23
   
   # Cycle operations with functional style
   odd_k_values = list(P(281).cycle(map=F.k(), filter=F.isodd))

Installation
------------

.. code-block:: bash

   pip install plumial

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   mathematical_foundations
   api_reference
   examples

