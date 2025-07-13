API Reference
=============

This section provides a complete reference for all classes, functions, and modules in the Plumial package.

Core Classes
------------

.. automodule:: plumial.core
   :members:
   :undoc-members:
   :show-inheritance:

Core Implementation Details
---------------------------

P Class (Path Polynomials)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Factory Function
^^^^^^^^^^^^^^^^

.. autofunction:: plumial.core.P.P

Path Object Methods
^^^^^^^^^^^^^^^^^^^

The ``P()`` function returns a path object with the following methods:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plumial.core.P._P

.. toctree::
   :maxdepth: 1

   generated/plumial.core.P._P.k
   generated/plumial.core.P._P.a
   generated/plumial.core.P._P.x
   generated/plumial.core.P._P.f
   generated/plumial.core.P._P.encode
   generated/plumial.core.P._P.cycle
   generated/plumial.core.P._P.uv
   generated/plumial.core.P._P.next
   generated/plumial.core.P._P.pred
   generated/plumial.core.P._P.p
   generated/plumial.core.P._P.n
   generated/plumial.core.P._P.o
   generated/plumial.core.P._P.e
   generated/plumial.core.P._P.basis
   generated/plumial.core.P._P.isforced
   generated/plumial.core.P._P.b
   generated/plumial.core.P._P.ax
   generated/plumial.core.P._P.D

.. autosummary::
   :toctree: generated/

   plumial.core.P._P.d
   plumial.core.P._P.k
   plumial.core.P._P.a
   plumial.core.P._P.x
   plumial.core.P._P.f
   plumial.core.P._P.encode
   plumial.core.P._P.cycle
   plumial.core.P._P.uv
   plumial.core.P._P.next
   plumial.core.P._P.pred
   plumial.core.P._P.p
   plumial.core.P._P.n
   plumial.core.P._P.o
   plumial.core.P._P.e
   plumial.core.P._P.basis
   plumial.core.P._P.isforced
   plumial.core.P._P.b
   plumial.core.P._P.ax
   plumial.core.P._P.D

.. autoclass:: plumial.core.P._P
   :members: d, k, a, x, f, encode, cycle, uv, next, pred, p, n, o, e, basis, isforced, b, ax, D
   :undoc-members:
   :show-inheritance:

D Class (Difference Polynomials)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Factory Function
^^^^^^^^^^^^^^^^

.. autofunction:: plumial.core.D.D

D Object Methods
^^^^^^^^^^^^^^^^

The ``D()`` function returns a d-polynomial object with the following methods:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plumial.core.D._D

.. toctree::
   :maxdepth: 1

   generated/plumial.core.D._D.d
   generated/plumial.core.D._D.encode
   generated/plumial.core.D._D.as_expr
   generated/plumial.core.D._D.n
   generated/plumial.core.D._D.o
   generated/plumial.core.D._D.e
   generated/plumial.core.D._D.basis

.. autosummary::
   :toctree: generated/

   plumial.core.D._D.d
   plumial.core.D._D.encode
   plumial.core.D._D.as_expr
   plumial.core.D._D.n
   plumial.core.D._D.o
   plumial.core.D._D.e
   plumial.core.D._D.basis

.. autoclass:: plumial.core.D._D
   :members: d, encode, as_expr, n, o, e, basis
   :undoc-members:
   :show-inheritance:

Basis System
~~~~~~~~~~~~

.. automodule:: plumial.core.basis
   :members:
   :undoc-members:
   :show-inheritance:

Utility Modules
---------------

Symbolic Variables
~~~~~~~~~~~~~~~~~~

.. automodule:: plumial.utils.symbolic
   :members:
   :undoc-members:
   :show-inheritance:

Functions Module
~~~~~~~~~~~~~~~~

.. automodule:: plumial.utils.functions
   :members:
   :undoc-members:
   :show-inheritance:

Binary Operations
~~~~~~~~~~~~~~~~~

.. automodule:: plumial.utils.binary
   :members:
   :undoc-members:
   :show-inheritance:

Matrix Utilities
~~~~~~~~~~~~~~~~

.. automodule:: plumial.utils.matrix_utils
   :members:
   :undoc-members:
   :show-inheritance:

Type System
-----------

.. automodule:: plumial.types
   :members:
   :undoc-members:
   :show-inheritance: