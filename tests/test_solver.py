"""
Test suite for Sudoku Graph Coloring solver
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sudoku import Sudoku
from solver import SudokuSolver
from puzzles import get_puzzle_by_name


class TestSudokuSolver(unittest.TestCase):
    """Test cases for Sudoku solver"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Load test puzzles
        self.easy_puzzle_data = get_puzzle_by_name("very_easy")
        self.hard_puzzle_data = get_puzzle_by_name("hard")
        
        # Create Sudoku objects
        self.easy_sudoku = Sudoku()
        self.easy_sudoku.load_from_list(self.easy_puzzle_data['puzzle'])
        
        self.hard_sudoku = Sudoku()
        self.hard_sudoku.load_from_list(self.hard_puzzle_data['puzzle'])
    
    def test_solver_initialization(self):
        """Test solver initialization"""
        solver = SudokuSolver(self.easy_sudoku)
        
        self.assertIsNotNone(solver.sudoku)
        self.assertIsNotNone(solver.graph)
        self.assertEqual(solver.steps, 0)
        self.assertEqual(solver.backtracks, 0)
    
    def test_backtracking_easy(self):
        """Test backtracking on easy puzzle"""
        solver = SudokuSolver(self.easy_sudoku)
        success = solver.solve_backtracking(use_heuristics=False)
        
        self.assertTrue(success)
        self.assertTrue(self.easy_sudoku.is_solved())
        self.assertTrue(self.easy_sudoku.is_valid_solution())
        self.assertGreater(solver.steps, 0)
    
    def test_backtracking_with_heuristics_easy(self):
        """Test backtracking with heuristics on easy puzzle"""
        solver = SudokuSolver(self.easy_sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        self.assertTrue(success)
        self.assertTrue(self.easy_sudoku.is_solved())
        self.assertTrue(self.easy_sudoku.is_valid_solution())
    
    def test_forward_checking_easy(self):
        """Test forward checking on easy puzzle"""
        solver = SudokuSolver(self.easy_sudoku)
        success = solver.solve_forward_checking()
        
        self.assertTrue(success)
        self.assertTrue(self.easy_sudoku.is_solved())
        self.assertTrue(self.easy_sudoku.is_valid_solution())
    
    def test_arc_consistency_easy(self):
        """Test arc consistency on easy puzzle"""
        solver = SudokuSolver(self.easy_sudoku)
        success = solver.solve_arc_consistency()
        
        self.assertTrue(success)
        self.assertTrue(self.easy_sudoku.is_solved())
        self.assertTrue(self.easy_sudoku.is_valid_solution())
    
    def test_backtracking_hard(self):
        """Test backtracking on hard puzzle"""
        solver = SudokuSolver(self.hard_sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        self.assertTrue(success)
        self.assertTrue(self.hard_sudoku.is_solved())
        self.assertTrue(self.hard_sudoku.is_valid_solution())
    
    def test_solution_statistics(self):
        """Test solution statistics collection"""
        solver = SudokuSolver(self.easy_sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        self.assertTrue(success)
        
        stats = solver.get_solution_statistics()
        
        self.assertIn('algorithm', stats)
        self.assertIn('steps', stats)
        self.assertIn('backtracks', stats)
        self.assertIn('solving_time', stats)
        self.assertIn('solution_found', stats)
        self.assertIn('is_valid', stats)
        
        self.assertTrue(stats['solution_found'])
        self.assertTrue(stats['is_valid'])
        self.assertGreater(stats['steps'], 0)
    
    def test_solve_sudoku_convenience(self):
        """Test the convenience solve_sudoku function"""
        from solver import solve_sudoku
        
        success, solver = solve_sudoku(self.easy_sudoku, "backtracking_heuristics")
        
        self.assertTrue(success)
        self.assertTrue(self.easy_sudoku.is_solved())
        self.assertTrue(self.easy_sudoku.is_valid_solution())
        self.assertIsInstance(solver, SudokuSolver)
    
    def test_invalid_algorithm(self):
        """Test with invalid algorithm name"""
        from solver import solve_sudoku
        
        success, solver = solve_sudoku(self.easy_sudoku, "invalid_algorithm")
        
        self.assertFalse(success)
    
    def test_empty_puzzle(self):
        """Test solving empty puzzle (should find a solution)"""
        empty_puzzle_data = get_puzzle_by_name("empty")
        empty_sudoku = Sudoku()
        empty_sudoku.load_from_list(empty_puzzle_data['puzzle'])
        
        solver = SudokuSolver(empty_sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        # Empty puzzle should have MANY solutions
        self.assertTrue(success)
        self.assertTrue(empty_sudoku.is_solved())
        self.assertTrue(empty_sudoku.is_valid_solution())
    
    def test_already_solved_puzzle(self):
        """Test solving already solved puzzle"""
        solved_puzzle_data = get_puzzle_by_name("solved")
        solved_sudoku = Sudoku()
        solved_sudoku.load_from_list(solved_puzzle_data['puzzle'])
        
        solver = SudokuSolver(solved_sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        self.assertTrue(success)
        self.assertTrue(solved_sudoku.is_solved())
        self.assertTrue(solved_sudoku.is_valid_solution())
        # Steps should be minimal since it's already solved
        self.assertLessEqual(solver.steps, 10)
    
    def test_graph_integration(self):
        """Test that graph colors match Sudoku grid after solving"""
        solver = SudokuSolver(self.easy_sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        self.assertTrue(success)
        
        # Check that graph colors match Sudoku grid
        for v in range(81):
            row, col = divmod(v, 9)
            graph_color = solver.graph.colors[v]
            grid_value = self.easy_sudoku.get_cell(row, col)
            
            if graph_color is None:
                self.assertEqual(grid_value, 0)
            else:
                self.assertEqual(graph_color, grid_value)
    
    def test_solution_history(self):
        """Test that solution history is recorded"""
        solver = SudokuSolver(self.easy_sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        self.assertTrue(success)
        self.assertGreater(len(solver.solution_history), 0)
        
        # Check history structure
        for state in solver.solution_history:
            self.assertIn('step', state)
            self.assertIn('grid', state)
            self.assertIn('description', state)
            self.assertIn('graph_stats', state)
            
            self.assertIsInstance(state['grid'], list)
            self.assertEqual(len(state['grid']), 9)
            self.assertEqual(len(state['grid'][0]), 9)


def run_all_tests():
    """Run all tests and print results"""
    print("Running Sudoku Solver Tests...")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSudokuSolver)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")
        
        # Print failure details
        for test, traceback in result.failures:
            print(f"\nFailure in {test}:")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\nError in {test}:")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    
    # Run a quick demo if tests pass
    if success:
        print("\n" + "="*60)
        print("QUICK DEMO")
        print("="*60)
        
        # Import here to avoid circular imports
        from examples.puzzles import get_puzzle_by_name
        
        # Test solving a puzzle
        puzzle_data = get_puzzle_by_name("medium")
        sudoku = Sudoku()
        sudoku.load_from_list(puzzle_data['puzzle'])
        
        print(f"\nSolving {puzzle_data['name']}...")
        print(f"Difficulty: {puzzle_data['difficulty']}")
        print(f"Empty cells: {len(sudoku.get_empty_cells())}")
        
        solver = SudokuSolver(sudoku)
        success = solver.solve_backtracking(use_heuristics=True)
        
        if success:
            print("\n✅ Puzzle solved successfully!")
            print(f"Steps: {solver.steps}")
            print(f"Backtracks: {solver.backtracks}")
            print(f"Time: {solver.solving_time:.3f}s")
            
            # Verify solution
            if sudoku.is_valid_solution():
                print("🎯 Solution is valid!")
            else:
                print("❌ Solution is invalid!")
        else:
            print("❌ Failed to solve puzzle")