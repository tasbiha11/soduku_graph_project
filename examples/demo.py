#!/usr/bin/env python3
"""
Sudoku Graph Coloring Demo
Interactive demonstration of Sudoku solving using graph coloring
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sudoku import Sudoku, create_example_sudoku
from solver import SudokuSolver, solve_sudoku
from visualizer import SudokuVisualizer
from utils import generate_random_sudoku, analyze_puzzle, benchmark_solvers
from puzzles import get_puzzle_by_name, list_all_puzzles, print_puzzle_info


def print_banner():
    """Print program banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║       SUDOKU SOLVER USING GRAPH COLORING                    ║
    ║       A Constraint Satisfaction Problem Demo                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This project demonstrates how Sudoku can be solved using graph theory:
    - Each cell is a vertex (81 vertices total)
    - Constraint edges connect cells in same row, column, or 3x3 box
    - Solving Sudoku = Coloring graph with 9 colors (1-9)
    - No adjacent vertices can share the same color
    
    Algorithms implemented:
    • Backtracking with MRV & LCV heuristics
    • Forward Checking
    • Arc Consistency (AC-3)
    • Brute Force (for comparison)
    """
    print(banner)


def demo_basic_solving():
    """Demonstrate basic Sudoku solving"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Sudoku Solving with Graph Coloring")
    print("="*70)
    
    # Create a puzzle
    sudoku = create_example_sudoku()
    
    print("\n📊 Original Sudoku Puzzle:")
    print(sudoku)
    
    # Create visualizer
    visualizer = SudokuVisualizer(sudoku)
    
    # Show initial visualizations
    print("\n🖼️  Visualizing puzzle and constraint graph...")
    visualizer.plot_side_by_side("Sudoku Puzzle and Constraint Graph")
    visualizer.show()
    
    # Solve using backtracking with heuristics
    print("\n⚙️  Solving with Backtracking + Heuristics (MRV + LCV)...")
    
    solver = SudokuSolver(sudoku)
    start_time = time.time()
    success = solver.solve_backtracking(use_heuristics=True)
    solve_time = time.time() - start_time
    
    if success:
        print("\n✅ Puzzle Solved Successfully!")
        print(f"\n📊 Solved Sudoku:")
        print(sudoku)
        
        # Verify solution
        if sudoku.is_valid_solution():
            print("🎯 Solution is valid!")
        else:
            print("❌ Solution is invalid!")
        
        # Show solving statistics
        solver.print_statistics()
        
        # Visualize solved puzzle
        visualizer.set_sudoku(sudoku)
        visualizer.plot_sudoku_grid("Solved Sudoku")
        visualizer.show()
    else:
        print("❌ Failed to solve puzzle")


def demo_algorithm_comparison():
    """Compare different solving algorithms"""
    print("\n" + "="*70)
    print("DEMO 2: Algorithm Comparison")
    print("="*70)
    
    print("\n🔬 Comparing different graph coloring algorithms...")
    
    # Use a medium difficulty puzzle
    from puzzles import get_puzzle_by_name
    puzzle_data = get_puzzle_by_name("medium")
    
    sudoku = Sudoku()
    sudoku.load_from_list(puzzle_data['puzzle'])
    sudoku.difficulty = puzzle_data['difficulty']
    
    print(f"\n📊 Test Puzzle: {puzzle_data['name']}")
    print(f"Difficulty: {puzzle_data['difficulty']}")
    print(f"Empty cells: {len(sudoku.get_empty_cells())}")
    
    # Create solver
    solver = SudokuSolver(sudoku)
    
    # Compare algorithms
    algorithms = ["backtracking", "backtracking_heuristics", 
                 "forward_checking", "arc_consistency"]
    
    results = solver.compare_algorithms(algorithms)
    
    # Visualize comparison
    visualizer = SudokuVisualizer()
    visualizer.plot_algorithm_comparison(results)


def demo_solving_animation():
    """Demonstrate solving process with animation"""
    print("\n" + "="*70)
    print("DEMO 3: Solving Process Animation")
    print("="*70)
    
    print("\n🎥 Animating the solving process...")
    print("This will show step-by-step how the algorithm works.")
    
    # Use a smaller puzzle for faster demonstration
    puzzle_str = "000260701680700090190004500820100040004602900050003028009300074040050036703018000"
    
    sudoku = Sudoku()
    sudoku.load_from_string(puzzle_str)
    
    print("\n📊 Puzzle to solve:")
    print(sudoku)
    
    # Create solver with detailed history recording
    solver = SudokuSolver(sudoku)
    
    print("\n⚙️  Solving with Backtracking + Heuristics...")
    success = solver.solve_backtracking(use_heuristics=True)
    
    if success:
        print(f"\n✅ Solved in {solver.steps} steps with {solver.backtracks} backtracks")
        
        # Create animation
        visualizer = SudokuVisualizer(sudoku)
        
        print("\n🎬 Creating animation of solving process...")
        print("Note: Animation may take a moment to render.")
        
        # We'll animate a subset of steps for performance
        if len(solver.solution_history) > 100:
            # Sample every nth step
            sampled_history = []
            step = max(1, len(solver.solution_history) // 50)
            for i in range(0, len(solver.solution_history), step):
                sampled_history.append(solver.solution_history[i])
            
            # Add final state
            if sampled_history[-1] != solver.solution_history[-1]:
                sampled_history.append(solver.solution_history[-1])
            
            solver.solution_history = sampled_history
        
        # Create animation
        visualizer.animate_solving(solver, interval=200)
        
        # Also show solving progress plot
        visualizer.plot_solving_progress(solver)
    else:
        print("❌ Failed to solve puzzle")


def demo_graph_visualization():
    """Demonstrate graph visualization features"""
    print("\n" + "="*70)
    print("DEMO 4: Advanced Graph Visualization")
    print("="*70)
    
    print("\n🖼️  Exploring the constraint graph representation...")
    
    # Create a puzzle
    sudoku = create_example_sudoku()
    
    # Create visualizer
    visualizer = SudokuVisualizer(sudoku)
    
    # Show different visualizations
    print("\n1. Sudoku Grid with Candidates")
    visualizer.plot_sudoku_grid("Sudoku with Candidate Numbers", show_candidates=True)
    visualizer.show()
    
    print("\n2. Constraint Graph (Full View)")
    visualizer.plot_graph_representation("Sudoku Constraint Graph")
    visualizer.show()
    
    print("\n3. Side-by-Side Comparison")
    visualizer.plot_side_by_side()
    visualizer.show()
    
    # Analyze the graph
    if sudoku.graph:
        stats = sudoku.graph.get_graph_statistics()
        print("\n📈 Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show conflicts if any
        conflicts = sudoku.graph.get_color_conflicts()
        if conflicts:
            print(f"\n⚠️  Found {len(conflicts)} color conflicts")
            visualizer.plot_sudoku_grid("Conflicts Highlighted", highlight_conflicts=True)
            visualizer.show()


def demo_puzzle_generation():
    """Demonstrate puzzle generation and analysis"""
    print("\n" + "="*70)
    print("DEMO 5: Puzzle Generation and Analysis")
    print("="*70)
    
    print("\n🎲 Generating random Sudoku puzzles...")
    
    # Generate puzzles of different difficulties
    difficulties = ["easy", "medium", "hard"]
    
    for difficulty in difficulties:
        print(f"\nGenerating {difficulty} puzzle...")
        
        # Generate puzzle
        sudoku = generate_random_sudoku(difficulty)
        
        print(f"\n📊 Generated {difficulty.capitalize()} Puzzle:")
        print(sudoku)
        
        # Analyze puzzle
        analysis = analyze_puzzle(sudoku)
        print(f"\n📈 Analysis:")
        print(f"  Givens: {analysis['givens']}")
        print(f"  Empty cells: {analysis['empty_cells']}")
        print(f"  Difficulty: {analysis['difficulty']}")
        print(f"  Symmetric: {analysis['symmetric']}")
        
        # Try to solve it
        print(f"\n⚙️  Solving generated puzzle...")
        solver = SudokuSolver(sudoku)
        start_time = time.time()
        success = solver.solve_backtracking(use_heuristics=True)
        solve_time = time.time() - start_time
        
        if success:
            print(f"✅ Solved in {solve_time:.3f}s, {solver.steps} steps")
            
            # Verify solution
            if sudoku.is_valid_solution():
                print("🎯 Generated solution is valid!")
            else:
                print("❌ Generated solution is invalid!")
        else:
            print("❌ Failed to solve generated puzzle")
        
        print("-"*50)


def demo_interactive_solving():
    """Interactive solving demo"""
    print("\n" + "="*70)
    print("DEMO 6: Interactive Solving")
    print("="*70)
    
    print("\n🎮 Interactive Sudoku Solver")
    print("Choose a puzzle and watch it being solved step-by-step")
    
    # List available puzzles
    print("\n📚 Available Puzzles:")
    puzzles = list_all_puzzles()
    
    for i, puzzle_name in enumerate(puzzles[:6], 1):  # Show first 6
        puzzle_data = get_puzzle_by_name(puzzle_name)
        print(f"{i}. {puzzle_data['name']} ({puzzle_data['difficulty']})")
    
    print("7. Custom puzzle (enter your own)")
    
    choice = input("\nSelect puzzle (1-7): ").strip()
    
    sudoku = None
    
    if choice == "7":
        # Custom puzzle
        print("\nEnter your puzzle (81 characters, 0 or . for empty):")
        puzzle_str = input("> ").strip()
        
        if len(puzzle_str) != 81:
            print(f"Error: Expected 81 characters, got {len(puzzle_str)}")
            return
        
        sudoku = Sudoku()
        if not sudoku.load_from_string(puzzle_str):
            print("Error: Invalid puzzle string")
            return
    
    elif choice in ["1", "2", "3", "4", "5", "6"]:
        # Pre-defined puzzle
        puzzle_names = puzzles[:6]
        puzzle_name = puzzle_names[int(choice) - 1]
        puzzle_data = get_puzzle_by_name(puzzle_name)
        
        sudoku = Sudoku()
        sudoku.load_from_list(puzzle_data['puzzle'])
        sudoku.difficulty = puzzle_data['difficulty']
        
        print(f"\nSelected: {puzzle_data['name']}")
        print_puzzle_info(puzzle_name)
    
    else:
        print("Invalid choice")
        return
    
    if sudoku:
        # Show puzzle
        print("\n📊 Selected Puzzle:")
        print(sudoku)
        
        # Choose algorithm
        print("\n⚙️  Select Solving Algorithm:")
        print("1. Backtracking (basic)")
        print("2. Backtracking + Heuristics (MRV + LCV)")
        print("3. Forward Checking")
        print("4. Arc Consistency (AC-3)")
        
        algo_choice = input("\nSelect algorithm (1-4): ").strip()
        
        algorithms = {
            "1": "backtracking",
            "2": "backtracking_heuristics",
            "3": "forward_checking",
            "4": "arc_consistency"
        }
        
        if algo_choice in algorithms:
            algorithm = algorithms[algo_choice]
            
            print(f"\nSolving with {algorithm}...")
            
            # Solve
            success, solver = solve_sudoku(sudoku, algorithm)
            
            if success:
                print("\n✅ Puzzle Solved!")
                print(f"\n📊 Solved Sudoku:")
                print(sudoku)
                
                solver.print_statistics()
                
                # Ask if user wants to see visualization
                viz_choice = input("\nShow visualization? (y/n): ").strip().lower()
                if viz_choice == 'y':
                    visualizer = SudokuVisualizer(sudoku)
                    visualizer.plot_side_by_side("Solved Sudoku")
                    visualizer.show()
            else:
                print("\n❌ Failed to solve puzzle")
        else:
            print("Invalid algorithm choice")


def demo_benchmark():
    """Benchmark different algorithms"""
    print("\n" + "="*70)
    print("DEMO 7: Algorithm Benchmark")
    print("="*70)
    
    print("\n📊 Benchmarking different solving algorithms...")
    print("This may take a moment...")
    
    # Create test puzzles
    from utils import create_sample_puzzles
    samples = create_sample_puzzles()
    
    puzzles = list(samples.values())
    
    # Run benchmark
    from utils import benchmark_solvers
    results = benchmark_solvers(puzzles)
    
    print("\n📈 Benchmark Results Summary:")
    for algo, stats in results.items():
        print(f"\n{algo}:")
        print(f"  Average time: {stats['avg_time']:.4f}s")
        print(f"  Average steps: {stats['avg_steps']:.0f}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
    
    # Visualize results
    visualizer = SudokuVisualizer()
    visualizer.plot_algorithm_comparison(results)


def main_menu():
    """Main interactive menu"""
    print_banner()
    
    demos = {
        '1': ("Basic Sudoku Solving", demo_basic_solving),
        '2': ("Algorithm Comparison", demo_algorithm_comparison),
        '3': ("Solving Animation", demo_solving_animation),
        '4': ("Graph Visualization", demo_graph_visualization),
        '5': ("Puzzle Generation", demo_puzzle_generation),
        '6': ("Interactive Solving", demo_interactive_solving),
        '7': ("Algorithm Benchmark", demo_benchmark),
        'q': ("Quit", None)
    }
    
    while True:
        print("\n" + "="*70)
        print("MAIN MENU - Select a Demo")
        print("="*70)
        
        for key, (description, _) in demos.items():
            print(f"  {key}. {description}")
        
        choice = input("\nEnter your choice (1-7, q to quit): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Thank you for exploring Sudoku Graph Coloring!")
            print("Project by: [Your Name]")
            break
        
        if choice in demos and demos[choice][1]:
            try:
                demos[choice][1]()
            except KeyboardInterrupt:
                print("\n\n⚠️  Demo interrupted.")
            except Exception as e:
                print(f"\n❌ Error during demo: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n❌ Invalid choice. Please try again.")
        
        if choice != 'q':
            input("\nPress Enter to continue...")


def quick_start():
    """Quick start function for immediate testing"""
    print("Sudoku Graph Coloring - Quick Start")
    print("="*50)
    
    # Create and solve a simple puzzle
    sudoku = create_example_sudoku()
    
    print("\nPuzzle:")
    print(sudoku)
    
    print("\nSolving with Backtracking + Heuristics...")
    
    solver = SudokuSolver(sudoku)
    success = solver.solve_backtracking(use_heuristics=True)
    
    if success:
        print("\nSolved!")
        print(sudoku)
        solver.print_statistics()
    else:
        print("Failed to solve")
    
    return success


if __name__ == "__main__":
    try:
        # Check if user wants quick start
        if len(sys.argv) > 1 and sys.argv[1] == "--quick":
            quick_start()
        else:
            main_menu()
    
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()