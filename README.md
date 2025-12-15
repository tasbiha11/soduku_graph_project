# Sudoku Solver using Graph Coloring

A comprehensive implementation of Sudoku solving algorithms using graph theory and constraint satisfaction techniques. This project demonstrates how Sudoku can be modeled as a graph coloring problem and solved using various algorithms.

## Project Overview

This project implements Sudoku solving using graph coloring where:
- Each Sudoku cell is a **vertex** in a graph (81 vertices total)
- Constraints (same row, column, 3x3 box) are **edges** between vertices
- Solving Sudoku = **Graph coloring** with 9 colors (1-9)
- No adjacent vertices can share the same color (proper coloring)

### Key Features:
- **Graph Representation**: Visualize Sudoku as a constraint graph
- **Multiple Algorithms**: Backtracking, Forward Checking, Arc Consistency
- **Heuristics**: MRV (Minimum Remaining Values), LCV (Least Constraining Value)
- **Visualization**: Step-by-step solving animation
- **Comprehensive Testing**: Test suite with various puzzle difficulties
- **Interactive Demo**: Command-line interface with multiple demos
- **Benchmarking**: Compare algorithm performance

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sudoku-graph-coloring.git
cd sudoku-graph-coloring

# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python examples/demo.py

# Run quick test
python examples/demo.py --quick

# Run tests
python -m pytest tests/

## **How to Run:**

1. **Install dependencies:**
```bash
pip install -r requirements.txt

2. python examples/demo.py

3. python tests/test_solver.py

4. python examples/demo.py --quick
