Examples
========

Interactive Jupyter Notebook
-----------------------------

For comprehensive, interactive examples with beautiful mathematical rendering, please see our **Jupyter notebook**:

📓 **`View Examples Notebook on GitHub <https://github.com/wildducktheories/plumial/blob/main/examples.ipynb>`_**

The notebook includes:

* **Quick Start** - Basic usage of path objects and polynomial operations
* **Symbolic Mathematics** - Working with d polynomials, k polynomials, and uv transformations  
* **Functional Programming** - Using the F namespace for elegant cycle operations
* **Forced vs Unforced Analysis** - Critical for Collatz conjecture research
* **Data Analysis** - Converting cycles to pandas DataFrames for statistical analysis
* **Advanced Mathematics** - Polynomial arithmetic and relationship verification
* **Performance Examples** - Caching analysis and batch processing

Why Jupyter?
~~~~~~~~~~~~

The examples are provided as an interactive Jupyter notebook because:

* **Better Math Rendering**: Polynomial equations display beautifully with LaTeX formatting
* **Interactive Exploration**: Run, modify, and experiment with examples in real-time  
* **Rich Output**: DataFrames, plots, and mathematical expressions render properly
* **Comprehensive Coverage**: All features demonstrated with working code
* **Easy Testing**: Validate examples and experiment with different parameters

Quick Example
~~~~~~~~~~~~~

Here's a taste of what you'll find in the notebook:

.. code-block:: python

   from plumial import P
   from plumial.core import B
   from plumial.utils import S, F
   
   # Create path object and explore
   p = P(133)
   display("Binary:", p.b())           # '10000101'  
   display("D polynomial:", p.d())     # h**5 - g**2
   display("Evaluated:", p.encode(B.Collatz).d())  # 23
   
   # Analyze forced vs unforced cycles  
   display(f"P(9) forced: {P(9).isforced()}")    # False (unforced)
   display(f"P(291) forced: {P(291).isforced()}") # True (forced)
   
   # Functional operations on cycles
   collatz_p281 = P(281).encode(B.Collatz)
   odd_k_values = list(collatz_p281.cycle(map=F.k(), filter=F.isodd))

**→ `Open the full notebook <https://github.com/wildducktheories/plumial/blob/main/examples.ipynb>`_ for complete examples**