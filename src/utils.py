"""
Utility functions for Sudoku Graph Coloring project
Includes puzzle loading, saving, generation, and analysis
"""

import random
import json
import os
import time
from typing import List, Dict, Tuple, Optional, Set
import itertools

from .sudoku import Sudoku
from .graph import Graph, create_sudoku_graph


def load_puzzle_from_file(filename: str) -> Optional[Sudoku]:
    """
    Load Sudoku puzzle from a file
    
    Supported formats:
    - .txt: 81 characters (0 or . for empty, 1-9 for numbers)
    - .json: JSON format with grid array
    
    Args:
        filename: Path to puzzle file
        
    Returns:
        Sudoku object or None if error
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        return None
    
    sudoku = Sudoku()
    
    try:
        if filename.endswith('.txt'):
            with open(filename, 'r') as f:
                content = f.read().strip()
                # Remove any non-digit characters except '.' and newlines
                lines = content.split('\n')
                puzzle_str = ''.join(''.join(c for c in line if c in '0123456789.') 
                                    for line in lines)
                
                if len(puzzle_str) != 81:
                    print(f"Error: Expected 81 characters, got {len(puzzle_str)}")
                    return None
                
                if sudoku.load_from_string(puzzle_str):
                    print(f"Loaded puzzle from {filename}")
                    return sudoku
        
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
                
                if 'grid' in data:
                    grid = data['grid']
                    if sudoku.load_from_list(grid):
                        if 'difficulty' in data:
                            sudoku.difficulty = data['difficulty']
                        print(f"Loaded puzzle from {filename}")
                        return sudoku
                else:
                    print("Error: JSON file must contain 'grid' field")
                    return None
        
        else:
            print(f"Error: Unsupported file format: {filename}")
            return None
    
    except Exception as e:
        print(f"Error loading puzzle from {filename}: {e}")
        return None
    
    return None


def save_puzzle_to_file(sudoku: Sudoku, filename: str):
    """
    Save Sudoku puzzle to a file
    
    Args:
        sudoku: Sudoku puzzle to save
        filename: Output filename (.txt or .json)
    """
    try:
        if filename.endswith('.txt'):
            with open(filename, 'w') as f:
                # Save as 9 lines of 9 characters
                for i in range(9):
                    line = ''.join(str(sudoku.grid[i][j]) if sudoku.grid[i][j] != 0 else '.' 
                                  for j in range(9))
                    f.write(line + '\n')
            print(f"Saved puzzle to {filename}")
        
        elif filename.endswith('.json'):
            data = {
                'grid': sudoku.grid,
                'difficulty': sudoku.difficulty,
                'is_solved': sudoku.is_solved(),
                'is_valid': sudoku.is_valid_solution(),
                'empty_cells': len(sudoku.get_empty_cells())
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved puzzle to {filename}")
        
        else:
            print(f"Error: Unsupported file format: {filename}")
    
    except Exception as e:
        print(f"Error saving puzzle to {filename}: {e}")


def generate_random_sudoku(difficulty: str = "medium", 
                          symmetric: bool = True) -> Sudoku:
    """
    Generate a random Sudoku puzzle
    
    Args:
        difficulty: "easy", "medium", "hard", "expert"
        symmetric: Whether to generate symmetric puzzle
        
    Returns:
        Generated Sudoku puzzle
    """
    print(f"Generating {difficulty} Sudoku puzzle...")
    
    # Start with a solved puzzle
    solved = _generate_solved_sudoku()
    
    # Create a copy to remove numbers from
    puzzle = solved.copy()
    
    # Determine number of cells to remove based on difficulty
    if difficulty == "easy":
        cells_to_remove = random.randint(40, 45)
    elif difficulty == "medium":
        cells_to_remove = random.randint(46, 52)
    elif difficulty == "hard":
        cells_to_remove = random.randint(53, 58)
    else:  # expert
        cells_to_remove = random.randint(59, 64)
    
    # Get all cell positions
    all_positions = [(r, c) for r in range(9) for c in range(9)]
    
    if symmetric:
        # For symmetric puzzles, remove pairs of cells
        symmetric_positions = []
        for r in range(5):  # Only need first 5 rows for symmetry
            for c in range(9):
                symmetric_positions.append((r, c))
        
        random.shuffle(symmetric_positions)
        
        removed = 0
        for r, c in symmetric_positions:
            if removed >= cells_to_remove:
                break
            
            # Calculate symmetric position
            r_sym = 8 - r
            c_sym = 8 - c
            
            # Remove both cells
            if puzzle.get_cell(r, c) != 0:
                puzzle.set_cell(r, c, 0, validate=False)
                removed += 1
            
            if (r_sym, c_sym) != (r, c) and puzzle.get_cell(r_sym, c_sym) != 0:
                puzzle.set_cell(r_sym, c_sym, 0, validate=False)
                removed += 1
    
    else:
        # Remove cells randomly
        random.shuffle(all_positions)
        
        for r, c in all_positions[:cells_to_remove]:
            puzzle.set_cell(r, c, 0, validate=False)
    
    # Set initial state and difficulty
    puzzle.initial_state = [[puzzle.grid[r][c] for c in range(9)] for r in range(9)]
    puzzle.difficulty = difficulty.capitalize()
    
    print(f"Generated {difficulty} puzzle with {cells_to_remove} empty cells")
    return puzzle


def _generate_solved_sudoku() -> Sudoku:
    """Generate a random solved Sudoku puzzle"""
    # Start with an empty puzzle
    sudoku = Sudoku()
    
    # Fill diagonal 3x3 boxes (they don't interact)
    for box in range(3):
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        
        for i in range(3):
            for j in range(3):
                row = box * 3 + i
                col = box * 3 + j
                sudoku.set_cell(row, col, numbers[i * 3 + j], validate=False)
    
    # Solve the rest using backtracking
    from .solver import SudokuSolver
    solver = SudokuSolver(sudoku)
    solver.solve_backtracking(use_heuristics=True)
    
    return sudoku


def analyze_puzzle(sudoku: Sudoku) -> Dict:
    """
    Analyze Sudoku puzzle characteristics
    
    Args:
        sudoku: Puzzle to analyze
        
    Returns:
        Dictionary with analysis results
    """
    if sudoku.initial_state is None:
        sudoku.initial_state = [[sudoku.grid[r][c] for c in range(9)] for r in range(9)]
    
    # Count givens
    givens = 0
    for r in range(9):
        for c in range(9):
            if sudoku.initial_state[r][c] != 0:
                givens += 1
    
    # Count empty cells
    empty_cells = len(sudoku.get_empty_cells())
    
    # Analyze difficulty based on multiple factors
    difficulty_score = 0
    
    # Factor 1: Number of givens (fewer givens = harder)
    difficulty_score += (81 - givens) * 0.5
    
    # Factor 2: Average candidates per empty cell
    total_candidates = 0
    empty_with_candidates = 0
    
    for r, c in sudoku.get_empty_cells():
        candidates = sudoku.get_candidates(r, c)
        total_candidates += len(candidates)
        if candidates:
            empty_with_candidates += 1
    
    avg_candidates = total_candidates / empty_with_candidates if empty_with_candidates > 0 else 0
    difficulty_score += (9 - avg_candidates) * 2  # Fewer candidates = harder
    
    # Determine difficulty level
    if difficulty_score < 30:
        difficulty = "Easy"
    elif difficulty_score < 45:
        difficulty = "Medium"
    elif difficulty_score < 60:
        difficulty = "Hard"
    else:
        difficulty = "Expert"
    
    # Check for symmetry
    symmetric = True
    for r in range(9):
        for c in range(9):
            r_sym = 8 - r
            c_sym = 8 - c
            if sudoku.initial_state[r][c] != sudoku.initial_state[r_sym][c_sym]:
                symmetric = False
                break
        if not symmetric:
            break
    
    # Check if puzzle has unique solution (simplified check)
    # This is a simplified check - in practice would need to attempt solving
    has_unique = "Unknown"  # Would require full solving to determine
    
    return {
        'givens': givens,
        'empty_cells': empty_cells,
        'difficulty_score': round(difficulty_score, 2),
        'difficulty': difficulty,
        'avg_candidates_per_empty': round(avg_candidates, 2),
        'symmetric': symmetric,
        'has_unique_solution': has_unique,
        'is_valid': sudoku.is_valid_solution() if sudoku.is_solved() else "Not solved"
    }


def print_puzzle_analysis(sudoku: Sudoku):
    """Print detailed analysis of Sudoku puzzle"""
    analysis = analyze_puzzle(sudoku)
    
    print("\n" + "="*60)
    print("PUZZLE ANALYSIS")
    print("="*60)
    
    print(f"Givens: {analysis['givens']}")
    print(f"Empty cells: {analysis['empty_cells']}")
    print(f"Difficulty score: {analysis['difficulty_score']}")
    print(f"Estimated difficulty: {analysis['difficulty']}")
    print(f"Average candidates per empty cell: {analysis['avg_candidates_per_empty']}")
    print(f"Symmetric: {analysis['symmetric']}")
    print(f"Has unique solution: {analysis['has_unique_solution']}")
    
    if sudoku.is_solved():
        print(f"Solution valid: {analysis['is_valid']}")
    
    print("="*60)


def graph_to_dot(graph: Graph, filename: str, show_colors: bool = True):
    """
    Export graph to Graphviz DOT format
    
    Args:
        graph: Graph to export
        filename: Output filename
        show_colors: Whether to include color information
    """
    try:
        with open(filename, 'w') as f:
            f.write("graph SudokuConstraints {\n")
            f.write("  layout=neato;\n")
            f.write("  node [shape=circle, style=filled];\n")
            f.write("  edge [color=\"#888888\", penwidth=0.5];\n\n")
            
            # Write nodes
            for v in range(graph.n_vertices):
                if v in graph.vertex_positions:
                    x, y = graph.vertex_positions[v]
                    
                    # Node attributes
                    attrs = []
                    
                    if show_colors and graph.colors[v] is not None:
                        color_value = graph.colors[v]
                        if color_value in graph.available_colors:
                            color_index = (color_value - 1) % len(graph.available_colors)
                            # Use color palette
                            colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', 
                                     '#118AB2', '#073B4C', '#EF476F', '#7B5E7B', '#FF9A76']
                            fill_color = colors[color_index]
                            attrs.append(f'fillcolor="{fill_color}"')
                    
                    label = str(graph.colors[v]) if graph.colors[v] is not None else ""
                    if graph.vertex_labels and v in graph.vertex_labels:
                        label = f"{graph.vertex_labels[v]}\\n{label}"
                    
                    attrs.append(f'label="{label}"')
                    attrs.append(f'pos="{x},{y}!"')
                    
                    f.write(f'  v{v} [{", ".join(attrs)}];\n')
            
            f.write("\n")
            
            # Write edges (avoid duplicates)
            written_edges = set()
            for u in range(graph.n_vertices):
                for v in graph.adjacency_list[u]:
                    if u < v and (u, v) not in written_edges:
                        f.write(f"  v{u} -- v{v};\n")
                        written_edges.add((u, v))
            
            f.write("}\n")
        
        print(f"Graph exported to DOT format: {filename}")
    
    except Exception as e:
        print(f"Error exporting graph to DOT: {e}")


def benchmark_solvers(puzzles: List[Sudoku], 
                     algorithms: List[str] = None) -> Dict[str, Dict]:
    """
    Benchmark different solvers on multiple puzzles
    
    Args:
        puzzles: List of Sudoku puzzles to test
        algorithms: List of algorithm names to test
        
    Returns:
        Benchmark results
    """
    if algorithms is None:
        algorithms = ["backtracking", "backtracking_heuristics", 
                     "forward_checking", "arc_consistency"]
    
    results = {algo: {"times": [], "steps": [], "success": []} 
               for algo in algorithms}
    
    print(f"\nBenchmarking {len(puzzles)} puzzles with {len(algorithms)} algorithms...")
    print("="*80)
    
    for i, puzzle in enumerate(puzzles):
        print(f"\nPuzzle {i+1}/{len(puzzles)}: {puzzle.difficulty} "
              f"({len(puzzle.get_empty_cells())} empty cells)")
        
        for algo in algorithms:
            # Create fresh copy for each test
            puzzle_copy = puzzle.copy()
            from .solver import SudokuSolver
            solver = SudokuSolver(puzzle_copy)
            
            # Time the solving
            start_time = time.time()
            
            if algo == "backtracking":
                success = solver.solve_backtracking(use_heuristics=False)
            elif algo == "backtracking_heuristics":
                success = solver.solve_backtracking(use_heuristics=True)
            elif algo == "forward_checking":
                success = solver.solve_forward_checking()
            elif algo == "arc_consistency":
                success = solver.solve_arc_consistency()
            else:
                continue
            
            solve_time = time.time() - start_time
            
            # Store results
            results[algo]["times"].append(solve_time)
            results[algo]["steps"].append(solver.steps)
            results[algo]["success"].append(success)
            
            print(f"  {algo:<25} {solve_time:.4f}s ({solver.steps} steps, "
                  f"{'✓' if success else '✗'})")
    
    # Calculate statistics
    stats = {}
    for algo in algorithms:
        times = results[algo]["times"]
        steps = results[algo]["steps"]
        success = results[algo]["success"]
        
        if times:
            stats[algo] = {
                "avg_time": sum(times) / len(times),
                "avg_steps": sum(steps) / len(steps),
                "success_rate": sum(success) / len(success) * 100,
                "total_time": sum(times),
                "min_time": min(times),
                "max_time": max(times)
            }
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<25} {'Avg Time':<10} {'Avg Steps':<12} {'Success':<10} {'Total Time':<12}")
    print("-"*80)
    
    for algo, stat in stats.items():
        print(f"{algo:<25} {stat['avg_time']:.4f}s    {stat['avg_steps']:<10.0f}  "
              f"{stat['success_rate']:.1f}%       {stat['total_time']:.4f}s")
    
    print("="*80)
    
    return stats


def create_sample_puzzles() -> Dict[str, Sudoku]:
    """
    Create a collection of sample puzzles with different difficulties
    
    Returns:
        Dictionary of difficulty -> Sudoku puzzle
    """
    print("Creating sample puzzles...")
    
    puzzles = {}
    
    # Easy puzzle (many givens)
    easy_str = """
    534678912
    672195348
    198342567
    859761423
    426853791
    713924856
    961537284
    287419635
    345286179
    """
    puzzles['easy'] = Sudoku()
    puzzles['easy'].load_from_string(easy_str)
    puzzles['easy'].difficulty = "Easy"
    
    # Medium puzzle (from earlier example)
    medium_str = """
    530070000
    600195000
    098000060
    800060003
    400803001
    700020006
    060000280
    000419005
    000080079
    """
    puzzles['medium'] = Sudoku()
    puzzles['medium'].load_from_string(medium_str)
    puzzles['medium'].difficulty = "Medium"
    
    # Hard puzzle
    hard_str = """
    800000000
    003600000
    070090200
    050007000
    000045700
    000100030
    001000068
    008500010
    090000400
    """
    puzzles['hard'] = Sudoku()
    puzzles['hard'].load_from_string(hard_str)
    puzzles['hard'].difficulty = "Hard"
    
    # Expert puzzle (very few givens)
    expert_str = """
    000000000
    000003000
    000020000
    000700000
    000000006
    000000300
    000080000
    000400000
    000000000
    """
    puzzles['expert'] = Sudoku()
    puzzles['expert'].load_from_string(expert_str)
    puzzles['expert'].difficulty = "Expert"
    
    print("Created sample puzzles: Easy, Medium, Hard, Expert")
    return puzzles


def validate_sudoku_string(puzzle_str: str) -> Tuple[bool, str]:
    """
    Validate Sudoku string format
    
    Args:
        puzzle_str: Sudoku string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Clean string
    puzzle_str = ''.join(puzzle_str.split())
    
    # Check length
    if len(puzzle_str) != 81:
        return False, f"Expected 81 characters, got {len(puzzle_str)}"
    
    # Check characters
    for char in puzzle_str:
        if char not in '0123456789.':
            return False, f"Invalid character '{char}'"
    
    # Check if puzzle has at least 17 clues (minimum for unique solution)
    clues = sum(1 for char in puzzle_str if char in '123456789')
    if clues < 17:
        return False, f"Only {clues} clues (minimum 17 for unique solution)"
    
    # Try to load into Sudoku object
    sudoku = Sudoku()
    if not sudoku.load_from_string(puzzle_str):
        return False, "Failed to parse puzzle"
    
    # Check for obvious contradictions
    for r in range(9):
        for c in range(9):
            value = sudoku.get_cell(r, c)
            if value != 0 and not sudoku.is_valid_move(r, c, value):
                return False, f"Contradiction at row {r+1}, column {c+1}"
    
    return True, "Valid Sudoku puzzle"


def get_sudoku_templates() -> Dict[str, str]:
    """
    Get Sudoku puzzle templates
    
    Returns:
        Dictionary of template names to puzzle strings
    """
    templates = {
        "blank": "0" * 81,
        "diagonal": (
            "123456789" + 
            "456789123" + 
            "789123456" + 
            "234567891" + 
            "567891234" + 
            "891234567" + 
            "345678912" + 
            "678912345" + 
            "912345678"
        ),
        "test_easy": (
            "534678912" +
            "672195348" +
            "198342567" +
            "859761423" +
            "426853791" +
            "713924856" +
            "961537284" +
            "287419635" +
            "345286179"
        ),
        "test_medium": (
            "530070000" +
            "600195000" +
            "098000060" +
            "800060003" +
            "400803001" +
            "700020006" +
            "060000280" +
            "000419005" +
            "000080079"
        )
    }
    
    return templates


if __name__ == "__main__":
    # Test utility functions
    print("Testing Sudoku Utilities...")
    
    # Test puzzle generation
    print("\n1. Generating random puzzles:")
    for difficulty in ["easy", "medium", "hard"]:
        puzzle = generate_random_sudoku(difficulty)
        print(f"   {difficulty}: {len(puzzle.get_empty_cells())} empty cells")
    
    # Test sample puzzles
    print("\n2. Sample puzzles:")
    samples = create_sample_puzzles()
    for difficulty, puzzle in samples.items():
        print(f"   {difficulty}: {len(puzzle.get_empty_cells())} empty cells")
    
    # Test analysis
    print("\n3. Puzzle analysis:")
    puzzle = samples['medium']
    analysis = analyze_puzzle(puzzle)
    for key, value in analysis.items():
        print(f"   {key}: {value}")
    
    # Test validation
    print("\n4. Puzzle validation:")
    test_string = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    is_valid, message = validate_sudoku_string(test_string)
    print(f"   Test string valid: {is_valid}")
    print(f"   Message: {message}")
    
    print("\nUtility functions test complete!")