# Plumial: Collatz Conjecture Analysis

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://your-docs-url.com)
[![PyPI version](https://badge.fury.io/py/plumial.svg)](https://badge.fury.io/py/plumial)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Plumial is a powerful Python library for mathematical analysis of the Collatz conjecture using polynomial representations. It transforms the discrete dynamics of Collatz sequences into algebraic operations on polynomial spaces, enabling systematic analysis of cycle structures and their properties.

## ✨ Key Features

- **Path Objects (P class)**: Hydrated path identifiers that encode Collatz sequence paths
- **Polynomial Representations**: UV polynomials and k polynomials for algebraic analysis
- **Cycle Analysis**: Complete cycle navigation, detection, and mathematical properties
- **Symbolic Mathematics**: Full SymPy integration with comprehensive mathematical operations
- **Performance Optimized**: LRU caching and efficient algorithms for large-scale analysis
- **Type Safe**: Comprehensive type hints and modern Python practices

## 🌀 The Mystery Revealed

<div align="center">

![Plumial Animation](https://raw.githubusercontent.com/yourusername/plumial/main/docs/_static/watch-animation.gif)

*Witness the hypnotic σ₂₈₁(u,v) polynomial visualization - a glimpse into the hidden mathematical beauty of "glitched" Collatz cycles*

</div>

This mesmerizing animation reveals something remarkable: **p=281** represents a glitched Collatz cycle that challenges our understanding of the 3x+1 conjecture. The visualization shows σ₂₈₁(u,v) = u² + uv² + v⁴ evaluated with complex roots of unity, transforming discrete binary patterns into continuous geometric flows.

## 🔗 Mathematical Foundations

**[📖 Read the Complete Mathematical Foundations](docs/mathematical_foundations.html)**

Dive deep into the theoretical framework that powers Plumial:
- **UV-Polynomial Theory**: Bijection between natural numbers and algebraic forms
- **Cycle Element Identity**: The fundamental relationship x·d = a·k for cycle elements
- **Binary Decomposition**: How p = 2^(n_p) + Σb_{p,i}2^i encodes path structure
- **Successor Operations**: Bit rotation operations that preserve polynomial structure
- **Advanced Topics**: Cyclotomic connections, forced vs unforced cycles, and more

## 🚀 Quick Start

### Installation

```bash
pip install plumial
```

### Basic Usage

```python
from plumial import P
from plumial.core import B
from plumial.utils import S, I, F

# Create a polynomial for p-value 133
p = P(133)

# Get bit counts and binary representation
print(f"n={p.n()}, o={p.o()}, e={p.e()}")  # n=7, o=2, e=5
print(f"Binary: {p.b()}")                   # Binary: 10000101

# Work with polynomial representations
print(p.d())    # h**5 - g**2 (d-polynomial)
print(p.k())    # k polynomial (symbolic)
print(p.uv())   # UV polynomial representation

# Evaluate numerically - modern basis approach
print(p.encode(B.Collatz).d())              # Modern: 23

# Legacy approach still works
print(p.d(g=3, h=2))                        # Legacy: 23

# Cycle operations with functional style
odd_k_values = list(P(293).cycle(map=F.k(), filter=F.isodd))

# Binary string constructor
assert P(133) == P("10000101")               # Equivalent results
```

### Advanced Analysis

```python
# Explore the famous glitched cycle
p281 = P(281)
cycle = list(p281.cycle())
print(f"Cycle length: {len(cycle)}")
print(f"Sigma polynomial: {p281.uv()}")  # u**2 + u*v**2 + v**4

# Mathematical verification
for p in cycle:
    print(f"{p.p():3d}: forced={p.isforced()}")

# Symbolic mathematics
import sympy as sy
a, x = p.ax()  # Get reduced cycle polynomials
assert sy.expand(x * p.d()) == sy.expand(a * p.k())  # Verify identity
```

## 📚 Documentation

- **[Mathematical Foundations](docs/mathematical_foundations.html)** - Complete theoretical framework
- **[API Reference](docs/api_reference.html)** - Comprehensive function documentation
- **[Tutorial](docs/tutorial.html)** - Step-by-step learning guide
- **[Examples](docs/examples.html)** - Interactive Jupyter notebooks

## 🧮 Mathematical Capabilities

### Polynomial Representations
- **σ-polynomials**: Binary path encoding as σₚ(u,v) polynomials
- **k-polynomials**: Transformation polynomials for cycle analysis
- **d-polynomials**: d-polynomials - d(g,h) = h^e - g^o an important term in the cycle element identity

### Cycle Analysis
- **Complete cycle enumeration** with efficient navigation
- **Forced vs unforced cycle classification**
- **Cycle element identity verification**: x·d = a·k relationships
- **Multiple Collatz variants**: (3x+1,x/2), (5x+1,x/2), (7x+1,x/2), etc.

### Advanced Mathematics
- **Cyclotomic polynomial factorization** for d-polynomials
- **Matrix representations** for polynomial manipulation
- **GCD analysis** and solution theory for cycle constraints
- **Binary operations** with complete bit-level analysis

## 🔬 Research Applications

Plumial enables systematic investigation of:
- **Cycle existence theorems** through polynomial constraint analysis
- **Uniqueness proofs** using GH-form canonical representations
- **Statistical analysis** of cycle length distributions
- **Visualization** of polynomial surfaces and cycle behavior

## 🛠 Development Installation

```bash
git clone https://github.com/yourusername/plumial.git
cd plumial
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest                    # Run all tests
pytest tests/test_p_class.py  # Run specific test file
```

### Building Documentation

```bash
cd docs
make html                 # HTML documentation
make latexpdf            # PDF documentation
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [SymPy](https://www.sympy.org/) for symbolic mathematics
- Documentation powered by [Sphinx](https://www.sphinx-doc.org/)
- Inspired by decades of Collatz conjecture research

---

*"The polynomial framework transforms Collatz analysis from computational iteration to algebraic constraint solving, revealing deep mathematical structures while maintaining computational accessibility."*