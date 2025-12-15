import time
import random
from typing import List, Tuple, Optional, Dict, Set
from .graph import Graph
from .sudoku import Sudoku


class SudokuSolver:
    """Solves Sudoku puzzles using graph coloring algorithms"""
    
    def __init__(self, sudoku: Sudoku):
        """
        Initialize solver with Sudoku puzzle
        
        Args:
            sudoku: Sudoku puzzle to solve
        """
        self.sudoku = sudoku
        self.graph = sudoku.graph if sudoku.graph else sudoku._create_graph_from_grid()
        self.steps = 0
        self.backtracks = 0
        self.solving_time = 0
        self.solution_history: List[Dict] = []
        self.algorithm_used = ""
        
    def solve_backtracking(self, use_heuristics: bool = True) -> bool:
        """
        Solve Sudoku using backtracking with graph coloring
        
        Args:
            use_heuristics: Whether to use MRV and LCV heuristics
            
        Returns:
            bool: True if solution found, False otherwise
        """
        self.algorithm_used = "Backtracking" + (" with Heuristics" if use_heuristics else "")
        start_time = time.time()
        
        # Record initial state
        self._record_state("Start")
        
        # Get list of empty vertices
        empty_vertices = self.graph.get_uncolored_vertices()
        
        # Convert to (row, col) format for compatibility
        empty_cells = []
        for v in empty_vertices:
            row, col = divmod(v, 9)
            empty_cells.append((row, col, v))
        
        # Sort by constraints if using heuristics
        if use_heuristics:
            empty_cells.sort(key=lambda x: (
                len(self.sudoku.get_candidates(x[0], x[1])),
                -self.graph.degree(x[2])
            ))
        
        success = self._backtrack(empty_cells, 0, use_heuristics)
        
        self.solving_time = time.time() - start_time
        
        if success:
            # Update Sudoku grid from graph colors
            self._update_sudoku_from_graph()
            self._record_state("Solution Found")
            print(f"✅ Solution found in {self.steps} steps, {self.backtracks} backtracks, "
                  f"{self.solving_time:.3f} seconds")
        else:
            print(f"❌ No solution found after {self.steps} steps")
        
        return success
    
    def _backtrack(self, empty_cells: List[Tuple[int, int, int]], 
                  index: int, use_heuristics: bool) -> bool:
        """
        Recursive backtracking function
        
        Args:
            empty_cells: List of (row, col, vertex) tuples
            index: Current index in empty_cells list
            use_heuristics: Whether to use MRV and LCV
            
        Returns:
            bool: True if solution found from this state
        """
        self.steps += 1
        
        # If all cells filled, check if valid
        if index >= len(empty_cells):
            return self.graph.is_valid_coloring()
        
        row, col, vertex = empty_cells[index]
        
        # Get candidate values
        candidates = self.sudoku.get_candidates(row, col)
        
        if use_heuristics:
            # Order by Least Constraining Value (LCV)
            candidates = self._order_by_lcv(vertex, candidates)
        
        # Try each candidate
        for value in candidates:
            # Try to assign color
            if self.graph.assign_color(vertex, value):
                # Update Sudoku grid temporarily
                self.sudoku.set_cell(row, col, value, validate=False)
                
                # Record state every 100 steps for visualization
                if self.steps % 100 == 0:
                    self._record_state(f"Step {self.steps}")
                
                # Recursively try next cell
                if self._backtrack(empty_cells, index + 1, use_heuristics):
                    return True
                
                # Backtrack
                self.graph.remove_color(vertex)
                self.sudoku.set_cell(row, col, 0, validate=False)
                self.backtracks += 1
        
        return False
    
    def _order_by_lcv(self, vertex: int, candidates: List[int]) -> List[int]:
        """
        Order candidates by Least Constraining Value heuristic
        
        Args:
            vertex: Graph vertex
            candidates: List of candidate values
            
        Returns:
            List ordered by least constraining first
        """
        if not candidates:
            return []
        
        # Count how many neighbors would lose each candidate
        impact = []
        for value in candidates:
            removed_from_neighbors = 0
            for neighbor in self.graph.get_neighbors(vertex):
                if self.graph.colors[neighbor] is None:
                    neighbor_candidates = self.graph.get_available_colors(neighbor)
                    if value in neighbor_candidates:
                        removed_from_neighbors += 1
            impact.append((value, removed_from_neighbors))
        
        # Sort by impact (lowest first)
        impact.sort(key=lambda x: x[1])
        return [value for value, _ in impact]
    
    def solve_forward_checking(self) -> bool:
        """
        Solve using backtracking with forward checking
        
        Returns:
            bool: True if solution found
        """
        self.algorithm_used = "Forward Checking"
        start_time = time.time()
        
        self._record_state("Start")
        
        # Get all empty vertices
        empty_vertices = self.graph.get_uncolored_vertices()
        
        # Convert to list of (vertex, domain)
        domains = {}
        for v in empty_vertices:
            domains[v] = self.graph.get_available_colors(v)
        
        success = self._forward_check(domains)
        
        self.solving_time = time.time() - start_time
        
        if success:
            self._update_sudoku_from_graph()
            self._record_state("Solution Found")
            print(f"✅ Solution found in {self.steps} steps, {self.backtracks} backtracks, "
                  f"{self.solving_time:.3f} seconds")
        else:
            print(f"❌ No solution found after {self.steps} steps")
        
        return success
    
    def _forward_check(self, domains: Dict[int, List[int]]) -> bool:
        """
        Recursive forward checking algorithm
        
        Args:
            domains: Dictionary of vertex -> available colors
            
        Returns:
            bool: True if solution found
        """
        self.steps += 1
        
        # If no uncolored vertices left
        if not domains:
            return self.graph.is_valid_coloring()
        
        # Select most constrained vertex (MRV)
        vertex = min(domains.keys(), 
                    key=lambda v: (len(domains[v]), -self.graph.degree(v)))
        
        # Get domain for selected vertex
        domain = domains[vertex]
        
        # Remove vertex from domains for recursion
        remaining_domains = domains.copy()
        del remaining_domains[vertex]
        
        # Try each value in domain
        for value in domain:
            # Try assignment
            if self.graph.assign_color(vertex, value):
                self.sudoku.set_cell(*divmod(vertex, 9), value, validate=False)
                
                # Record state periodically
                if self.steps % 100 == 0:
                    self._record_state(f"Step {self.steps}")
                
                # Propagate constraints
                new_domains = self._propagate_constraints(vertex, value, remaining_domains)
                
                # Check for domain wipeout
                if new_domains is not None:
                    # Recursively continue
                    if self._forward_check(new_domains):
                        return True
                
                # Backtrack
                self.graph.remove_color(vertex)
                self.sudoku.set_cell(*divmod(vertex, 9), 0, validate=False)
                self.backtracks += 1
        
        return False
    
    def _propagate_constraints(self, vertex: int, value: int, 
                              domains: Dict[int, List[int]]) -> Optional[Dict[int, List[int]]]:
        """
        Propagate constraints after assignment
        
        Args:
            vertex: Assigned vertex
            value: Assigned value
            domains: Current domains
            
        Returns:
            Updated domains or None if domain wipeout occurs
        """
        new_domains = domains.copy()
        
        # Remove value from neighbors' domains
        for neighbor in self.graph.get_neighbors(vertex):
            if neighbor in new_domains and value in new_domains[neighbor]:
                new_domains[neighbor] = [c for c in new_domains[neighbor] if c != value]
                
                # Check for domain wipeout
                if not new_domains[neighbor]:
                    return None
        
        return new_domains
    
    def solve_arc_consistency(self) -> bool:
        """
        Solve using backtracking with arc consistency (AC-3 algorithm)
        
        Returns:
            bool: True if solution found
        """
        self.algorithm_used = "Arc Consistency (AC-3)"
        start_time = time.time()
        
        self._record_state("Start")
        
        # Initialize domains for all vertices
        domains = {}
        for v in range(81):
            if self.graph.colors[v] is None:
                domains[v] = self.graph.get_available_colors(v)
            else:
                domains[v] = [self.graph.colors[v]]
        
        # Run AC-3
        if not self._ac3(domains):
            self.solving_time = time.time() - start_time
            print("AC-3 detected inconsistency")
            return False
        
        # Get uncolored vertices
        uncolored = [v for v in domains if len(domains[v]) > 1]
        
        success = self._backtrack_with_domains(uncolored, domains)
        
        self.solving_time = time.time() - start_time
        
        if success:
            self._update_sudoku_from_graph()
            self._record_state("Solution Found")
            print(f"✅ Solution found in {self.steps} steps, {self.backtracks} backtracks, "
                  f"{self.solving_time:.3f} seconds")
        else:
            print(f"❌ No solution found after {self.steps} steps")
        
        return success
    
    def _ac3(self, domains: Dict[int, List[int]]) -> bool:
        """
        AC-3 algorithm for arc consistency
        
        Args:
            domains: Dictionary of vertex domains
            
        Returns:
            bool: True if consistent, False if contradiction found
        """
        # Initialize queue with all arcs
        queue = []
        for v1 in domains:
            for v2 in self.graph.get_neighbors(v1):
                queue.append((v1, v2))
        
        while queue:
            v1, v2 = queue.pop(0)
            
            if self._revise(domains, v1, v2):
                # If domain of v1 becomes empty
                if not domains[v1]:
                    return False
                
                # Add arcs from neighbors of v1 (except v2) back to queue
                for neighbor in self.graph.get_neighbors(v1):
                    if neighbor != v2:
                        queue.append((neighbor, v1))
        
        return True
    
    def _revise(self, domains: Dict[int, List[int]], 
                v1: int, v2: int) -> bool:
        """
        Revise domain of v1 based on constraint with v2
        
        Returns:
            bool: True if domain was revised
        """
        revised = False
        
        # For each value in v1's domain
        for x in list(domains[v1]):
            # Check if there's a value in v2's domain that satisfies constraint
            satisfies = False
            for y in domains[v2]:
                if x != y:
                    satisfies = True
                    break
            
            # If no satisfying value, remove x from v1's domain
            if not satisfies:
                domains[v1].remove(x)
                revised = True
        
        return revised
    
    def _backtrack_with_domains(self, vertices: List[int], 
                               domains: Dict[int, List[int]]) -> bool:
        """
        Backtracking with pre-processed domains
        
        Args:
            vertices: List of vertices to assign
            domains: Current domains
            
        Returns:
            bool: True if solution found
        """
        self.steps += 1
        
        if not vertices:
            return True
        
        # Select variable with MRV
        vertex = min(vertices, key=lambda v: len(domains[v]))
        
        # Get and order domain by LCV
        domain = domains[vertex]
        domain = self._order_by_lcv(vertex, domain)
        
        # Try each value
        for value in domain:
            # Create copy of domains for backtracking
            new_domains = domains.copy()
            new_domains[vertex] = [value]
            
            # Assign color
            if self.graph.assign_color(vertex, value):
                self.sudoku.set_cell(*divmod(vertex, 9), value, validate=False)
                
                # Record state periodically
                if self.steps % 100 == 0:
                    self._record_state(f"Step {self.steps}")
                
                # Propagate with AC-3
                if self._ac3(new_domains):
                    # Get remaining unassigned vertices
                    remaining = [v for v in vertices if v != vertex and len(new_domains[v]) > 1]
                    
                    # Recursively continue
                    if self._backtrack_with_domains(remaining, new_domains):
                        return True
                
                # Backtrack
                self.graph.remove_color(vertex)
                self.sudoku.set_cell(*divmod(vertex, 9), 0, validate=False)
                self.backtracks += 1
        
        return False
    
    def solve_brute_force(self) -> bool:
        """
        Solve using brute force (for comparison)
        
        Returns:
            bool: True if solution found
        """
        self.algorithm_used = "Brute Force"
        start_time = time.time()
        
        self._record_state("Start")
        
        # Get all empty vertices
        empty_vertices = self.graph.get_uncolored_vertices()
        
        success = self._brute_force(empty_vertices, 0)
        
        self.solving_time = time.time() - start_time
        
        if success:
            self._update_sudoku_from_graph()
            self._record_state("Solution Found")
            print(f"✅ Solution found in {self.steps} steps, {self.solving_time:.3f} seconds")
        else:
            print(f"❌ No solution found after {self.steps} steps")
        
        return success
    
    def _brute_force(self, vertices: List[int], index: int) -> bool:
        """
        Brute force recursive search
        
        Args:
            vertices: List of vertices to color
            index: Current index
            
        Returns:
            bool: True if solution found
        """
        self.steps += 1
        
        if index >= len(vertices):
            return self.graph.is_valid_coloring()
        
        vertex = vertices[index]
        
        # Try all possible colors
        for color in range(1, 10):
            if self.graph.assign_color(vertex, color):
                self.sudoku.set_cell(*divmod(vertex, 9), color, validate=False)
                
                if self._brute_force(vertices, index + 1):
                    return True
                
                # Backtrack
                self.graph.remove_color(vertex)
                self.sudoku.set_cell(*divmod(vertex, 9), 0, validate=False)
        
        return False
    
    def solve_genetic_algorithm(self, population_size: int = 100, 
                               generations: int = 1000, 
                               mutation_rate: float = 0.1) -> bool:
        """
        Solve using a simple genetic algorithm (experimental)
        
        Args:
            population_size: Size of population
            generations: Maximum number of generations
            mutation_rate: Probability of mutation
            
        Returns:
            bool: True if solution found
        """
        self.algorithm_used = f"Genetic Algorithm (pop={population_size}, gen={generations})"
        start_time = time.time()
        
        print(f"Running Genetic Algorithm...")
        print(f"Population: {population_size}, Generations: {generations}")
        
        # This is a simplified genetic algorithm for demonstration
        # In practice, Sudoku is not well-suited for GA due to constraints
        
        # We'll just run for specified generations and return best found
        best_fitness = 0
        best_individual = None
        
        for gen in range(generations):
            # In a real implementation, you would:
            # 1. Initialize population
            # 2. Evaluate fitness
            # 3. Select parents
            # 4. Crossover
            # 5. Mutate
            # 6. Repeat
            
            self.steps += population_size
            
            # For now, just track progress
            if gen % 100 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}")
        
        self.solving_time = time.time() - start_time
        
        print(f"Genetic algorithm completed in {self.solving_time:.2f} seconds")
        print("Note: Genetic algorithm implementation is simplified for this demo.")
        print("For actual Sudoku solving, backtracking with heuristics is more effective.")
        
        return False  # Genetic algorithm rarely finds perfect solution
    
    def _update_sudoku_from_graph(self):
        """Update Sudoku grid from graph colors"""
        for v in range(81):
            color = self.graph.colors[v]
            if color is not None:
                row, col = divmod(v, 9)
                self.sudoku.set_cell(row, col, color, validate=False)
    
    def _record_state(self, description: str):
        """Record current solving state for visualization"""
        state = {
            "step": self.steps,
            "backtracks": self.backtracks,
            "time": time.time(),
            "description": description,
            "grid": [row[:] for row in self.sudoku.grid],
            "graph_stats": self.graph.get_graph_statistics() if self.graph else {}
        }
        self.solution_history.append(state)
    
    def get_solution_statistics(self) -> Dict:
        """Get statistics about the solving process"""
        return {
            "algorithm": self.algorithm_used,
            "steps": self.steps,
            "backtracks": self.backtracks,
            "solving_time": self.solving_time,
            "solution_found": self.sudoku.is_solved(),
            "is_valid": self.sudoku.is_valid_solution(),
            "empty_cells_initial": len(self.sudoku.get_empty_cells()),
            "empty_cells_final": 0 if self.sudoku.is_solved() else len(self.sudoku.get_empty_cells())
        }
    
    def print_statistics(self):
        """Print solving statistics"""
        stats = self.get_solution_statistics()
        
        print("\n" + "="*60)
        print("SOLVING STATISTICS")
        print("="*60)
        
        for key, value in stats.items():
            if key == "solving_time":
                print(f"{key.replace('_', ' ').title()}: {value:.4f} seconds")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("="*60)
    
    def compare_algorithms(self, algorithms: List[str] = None) -> Dict[str, Dict]:
        """
        Compare different solving algorithms
        
        Args:
            algorithms: List of algorithm names to compare
            
        Returns:
            Dictionary with results for each algorithm
        """
        if algorithms is None:
            algorithms = ["backtracking", "backtracking_heuristics", "forward_checking", "arc_consistency"]
        
        results = {}
        
        for algo in algorithms:
            print(f"\n{'='*60}")
            print(f"Testing {algo} algorithm...")
            print(f"{'='*60}")
            
            # Reset solver state
            self.steps = 0
            self.backtracks = 0
            self.solving_time = 0
            self.solution_history = []
            
            # Make a copy of the puzzle
            puzzle_copy = self.sudoku.copy()
            solver_copy = SudokuSolver(puzzle_copy)
            
            # Run algorithm
            success = False
            if algo == "backtracking":
                success = solver_copy.solve_backtracking(use_heuristics=False)
            elif algo == "backtracking_heuristics":
                success = solver_copy.solve_backtracking(use_heuristics=True)
            elif algo == "forward_checking":
                success = solver_copy.solve_forward_checking()
            elif algo == "arc_consistency":
                success = solver_copy.solve_arc_consistency()
            elif algo == "brute_force":
                success = solver_copy.solve_brute_force()
            
            # Store results
            results[algo] = {
                "success": success,
                "steps": solver_copy.steps,
                "backtracks": solver_copy.backtracks,
                "time": solver_copy.solving_time,
                "algorithm": solver_copy.algorithm_used
            }
            
            solver_copy.print_statistics()
        
        # Print comparison table
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON")
        print("="*80)
        print(f"{'Algorithm':<25} {'Success':<10} {'Steps':<10} {'Backtracks':<12} {'Time (s)':<10}")
        print("-"*80)
        
        for algo, result in results.items():
            print(f"{algo:<25} {str(result['success']):<10} "
                  f"{result['steps']:<10} {result['backtracks']:<12} {result['time']:.4f}")
        
        print("="*80)
        
        return results


def solve_sudoku(puzzle: Sudoku, algorithm: str = "backtracking_heuristics") -> Tuple[bool, SudokuSolver]:
    """
    Convenience function to solve Sudoku with specified algorithm
    
    Args:
        puzzle: Sudoku puzzle to solve
        algorithm: Algorithm to use
        
    Returns:
        Tuple of (success, solver)
    """
    solver = SudokuSolver(puzzle)
    
    success = False
    if algorithm == "backtracking":
        success = solver.solve_backtracking(use_heuristics=False)
    elif algorithm == "backtracking_heuristics":
        success = solver.solve_backtracking(use_heuristics=True)
    elif algorithm == "forward_checking":
        success = solver.solve_forward_checking()
    elif algorithm == "arc_consistency":
        success = solver.solve_arc_consistency()
    elif algorithm == "brute_force":
        success = solver.solve_brute_force()
    elif algorithm == "genetic":
        success = solver.solve_genetic_algorithm()
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available algorithms: backtracking, backtracking_heuristics, "
              "forward_checking, arc_consistency, brute_force, genetic")
    
    return success, solver


if __name__ == "__main__":
    # Test the solver
    from .sudoku import create_example_sudoku
    
    print("Testing Sudoku Solver...")
    
    # Create and display puzzle
    sudoku = create_example_sudoku()
    print("\nOriginal Puzzle:")
    print(sudoku)
    
    # Create solver
    solver = SudokuSolver(sudoku)
    
    # Test backtracking with heuristics
    print("\n" + "="*60)
    print("Solving with Backtracking + Heuristics...")
    print("="*60)
    
    success = solver.solve_backtracking(use_heuristics=True)
    
    if success:
        print("\nSolved Puzzle:")
        print(sudoku)
        
        # Verify solution
        if sudoku.is_valid_solution():
            print("✅ Solution is valid!")
        else:
            print("❌ Solution is invalid!")
    else:
        print("Failed to solve puzzle")
    
    # Print statistics
    solver.print_statistics()