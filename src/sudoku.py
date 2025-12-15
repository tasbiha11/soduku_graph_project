from typing import List, Optional, Tuple, Dict
import copy
from .graph import Graph, create_sudoku_graph


class Sudoku:
    """Represents a Sudoku puzzle"""
    
    def __init__(self, size: int = 9):
        """
        Initialize Sudoku puzzle
        
        Args:
            size: Size of Sudoku grid (default 9 for 9x9)
        """
        self.size = size
        self.box_size = int(size ** 0.5)  #3 for standard Sudoku
        self.grid: List[List[int]] = [[0 for _ in range(size)] for _ in range(size)]
        self.graph: Optional[Graph] = None
        self.initial_state: Optional[List[List[int]]] = None
        self.difficulty: str = "Unknown"
        
    def load_from_string(self, puzzle_str: str) -> bool:
        """
        Load puzzle from string representation
        
        Args:
            puzzle_str: String with 81 characters (0 for empty, 1-9 for filled)
            
        Returns:
            bool: True if successful, False otherwise
        """
        #Remove whitespace and newlines
        puzzle_str = ''.join(puzzle_str.split())
        
        #Standard Sudoku has 81 cells
        if len(puzzle_str) != 81:
            print(f"Error: Expected 81 characters, got {len(puzzle_str)}")
            return False
        
        #Parse string into grid
        index = 0
        for row in range(9):
            for col in range(9):
                char = puzzle_str[index]
                if char == '.' or char == '0':
                    self.grid[row][col] = 0
                elif '1' <= char <= '9':
                    self.grid[row][col] = int(char)
                else:
                    print(f"Error: Invalid character '{char}' at position {index}")
                    return False
                index += 1
        
        #Save initial state
        self.initial_state = copy.deepcopy(self.grid)
    
        #Create graph representation
        self._create_graph_from_grid()
        
        return True
    
    def load_from_list(self, puzzle_list: List[List[int]]) -> bool:
        """
        Load puzzle from 2D list
        
        Args:
            puzzle_list: 9x9 list of integers (0 for empty)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if len(puzzle_list) != 9 or any(len(row) != 9 for row in puzzle_list):
            print("Error: Puzzle must be 9x9")
            return False
        
        #Validate values
        for row in range(9):
            for col in range(9):
                value = puzzle_list[row][col]
                if not (0 <= value <= 9):
                    print(f"Error: Invalid value {value} at ({row}, {col})")
                    return False
        
        self.grid = copy.deepcopy(puzzle_list)
        self.initial_state = copy.deepcopy(self.grid)
        self._create_graph_from_grid()
        
        return True
    
    def _create_graph_from_grid(self):
        """Create graph representation from current grid"""
        self.graph = create_sudoku_graph()
        
        #Apply initial colors from grid
        for row in range(9):
            for col in range(9):
                vertex = row * 9 + col
                value = self.grid[row][col]
                if value != 0:
                    self.graph.assign_color(vertex, value)
    
    def get_cell(self, row: int, col: int) -> int:
        """Get value at specific cell"""
        return self.grid[row][col]
    
    def set_cell(self, row: int, col: int, value: int, validate: bool = True) -> bool:
        """
        Set value at specific cell
        
        Args:
            row, col: Cell coordinates (0-8)
            value: Value to set (0-9, 0 means empty)
            validate: Whether to validate the move
            
        Returns:
            bool: True if valid move, False otherwise
        """
        if not (0 <= row < 9 and 0 <= col < 9):
            return False
        
        if not (0 <= value <= 9):
            return False
        
        #Check if cell was initially filled
        if self.initial_state and self.initial_state[row][col] != 0 and value != self.initial_state[row][col]:
            print(f"Warning: Cell ({row}, {col}) was initially filled")
            return False
        
        #Validate move if requested
        if validate and value != 0:
            if not self.is_valid_move(row, col, value):
                return False
        
        #Update grid
        self.grid[row][col] = value
        
        #Update graph if it exists
        if self.graph:
            vertex = row * 9 + col
            if value == 0:
                self.graph.remove_color(vertex)
            else:
                self.graph.assign_color(vertex, value)
        
        return True
    
    def is_valid_move(self, row: int, col: int, value: int) -> bool:
        """
        Check if placing value at (row, col) is valid
        
        Args:
            row, col: Cell coordinates
            value: Value to check
            
        Returns:
            bool: True if move is valid
        """
        #Check row
        for c in range(9):
            if c != col and self.grid[row][c] == value:
                return False
        
        #Check column
        for r in range(9):
            if r != row and self.grid[r][col] == value:
                return False
        
        #Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r != row or c != col) and self.grid[r][c] == value:
                    return False
        
        return True
    
    def is_solved(self) -> bool:
        """Check if puzzle is completely and correctly solved"""
        #Check all cells are filled
        for row in self.grid:
            if 0 in row:
                return False
        
        #Check all constraints
        return self.is_valid_solution()
    
    def is_valid_solution(self) -> bool:
        """Check if current grid is a valid solution"""
        #Check rows
        for row in range(9):
            seen = set()
            for col in range(9):
                value = self.grid[row][col]
                if value == 0 or value in seen:
                    return False
                seen.add(value)
        
        #Check columns
        for col in range(9):
            seen = set()
            for row in range(9):
                value = self.grid[row][col]
                if value == 0 or value in seen:
                    return False
                seen.add(value)
        
        #Check 3x3 boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                seen = set()
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        value = self.grid[r][c]
                        if value == 0 or value in seen:
                            return False
                        seen.add(value)
        
        return True
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get list of empty cell coordinates"""
        empty = []
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    empty.append((row, col))
        return empty
    
    def get_candidates(self, row: int, col: int) -> List[int]:
        """
        Get possible values for a cell
        
        Args:
            row, col: Cell coordinates
            
        Returns:
            List of possible values (1-9)
        """
        if self.grid[row][col] != 0:
            return []
        
        candidates = []
        for value in range(1, 10):
            if self.is_valid_move(row, col, value):
                candidates.append(value)
        
        return candidates
    
    def get_difficulty_estimate(self) -> str:
        """
        Estimate puzzle difficulty based on number of givens
        
        Returns:
            Difficulty string (Easy, Medium, Hard, Expert)
        """
        if not self.initial_state:
            return "Unknown"
        
        givens = 0
        for row in range(9):
            for col in range(9):
                if self.initial_state[row][col] != 0:
                    givens += 1
        
        if givens >= 36:
            return "Easy"
        elif givens >= 32:
            return "Medium"
        elif givens >= 28:
            return "Hard"
        else:
            return "Expert"
    
    def to_string(self, show_grid: bool = True) -> str:
        """
        Convert puzzle to string representation
        
        Args:
            show_grid: Whether to show as grid or single line
            
        Returns:
            String representation
        """
        if show_grid:
            result = []
            result.append("+" + "---+" * 3)
            
            for row in range(9):
                line = "|"
                for col in range(9):
                    value = self.grid[row][col]
                    if value == 0:
                        line += "   "
                    else:
                        line += f" {value} "
                    
                    if (col + 1) % 3 == 0:
                        line += "|"
                
                result.append(line)
                
                if (row + 1) % 3 == 0:
                    result.append("+" + "---+" * 3)
            
            return "\n".join(result)
        else:
            #Single line representation
            result = []
            for row in range(9):
                for col in range(9):
                    value = self.grid[row][col]
                    result.append(str(value) if value != 0 else ".")
            return "".join(result)
    
    def to_graph_representation(self) -> Dict:
        """
        Convert to graph representation data
        
        Returns:
            Dictionary with graph data for visualization
        """
        if not self.graph:
            self._create_graph_from_grid()
        
        vertices = []
        edges = []
        colors = []
        
        for v in range(81):
            row, col = divmod(v, 9)
            x, y = self.graph.vertex_positions.get(v, (col, 8 - row))
            
            vertex_data = {
                "id": v,
                "label": self.graph.vertex_labels.get(v, f"R{row}C{col}"),
                "x": x,
                "y": y,
                "color": self.graph.colors.get(v),
                "row": row,
                "col": col,
                "is_initial": (self.initial_state[row][col] != 0 if self.initial_state else False)
            }
            vertices.append(vertex_data)
            
            if self.graph.colors[v] is not None:
                colors.append({
                    "vertex": v,
                    "color": self.graph.colors[v]
                })
        
        #Add edges (unique pairs)
        edge_set = set()
        for u in range(81):
            for v in self.graph.adjacency_list[u]:
                if u < v:  # Avoid duplicates
                    edge_set.add((u, v))
        
        for u, v in edge_set:
            edges.append({
                "from": u,
                "to": v,
                "id": f"{u}-{v}"
            })
        
        return {
            "vertices": vertices,
            "edges": edges,
            "colors": colors,
            "stats": self.graph.get_graph_statistics() if self.graph else {}
        }
    
    def copy(self) -> 'Sudoku':
        """Create a deep copy of the puzzle"""
        new_sudoku = Sudoku(self.size)
        new_sudoku.grid = copy.deepcopy(self.grid)
        new_sudoku.initial_state = copy.deepcopy(self.initial_state) if self.initial_state else None
        new_sudoku.difficulty = self.difficulty
        
        if self.graph:
            new_sudoku.graph = Graph(81)
            new_sudoku.graph.adjacency_list = copy.deepcopy(self.graph.adjacency_list)
            new_sudoku.graph.colors = copy.deepcopy(self.graph.colors)
            new_sudoku.graph.vertex_positions = copy.deepcopy(self.graph.vertex_positions)
            new_sudoku.graph.vertex_labels = copy.deepcopy(self.graph.vertex_labels)
        
        return new_sudoku
    
    def reset(self):
        """Reset puzzle to initial state"""
        if self.initial_state:
            self.grid = copy.deepcopy(self.initial_state)
            self._create_graph_from_grid()
    
    def __str__(self) -> str:
        """String representation"""
        return self.to_string()


def create_empty_sudoku() -> Sudoku:
    """Create an empty Sudoku puzzle"""
    sudoku = Sudoku()
    sudoku.initial_state = [[0 for _ in range(9)] for _ in range(9)]
    sudoku._create_graph_from_grid()
    return sudoku


def create_example_sudoku() -> Sudoku:
    """Create an example Sudoku puzzle (medium difficulty)"""
    puzzle_str = """
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
    
    sudoku = Sudoku()
    sudoku.load_from_string(puzzle_str)
    sudoku.difficulty = sudoku.get_difficulty_estimate()
    return sudoku


def print_sudoku_with_candidates(sudoku: Sudoku):
    """Print Sudoku grid with candidate numbers for empty cells"""
    print("\nSudoku with candidates:")
    print("+" + "-------------------+" * 3)
    
    for row in range(9):
        #Print three lines per row (for candidate display)
        for subrow in range(3):
            line = "|"
            for col in range(9):
                value = sudoku.get_cell(row, col)
                
                if value != 0:
                    if subrow == 1:
                        line += f"     {value}     "
                    else:
                        line += "           "
                else:
                    candidates = sudoku.get_candidates(row, col)
                    # Display candidates in 3x3 mini-grid
                    for subcol in range(3):
                        num = subrow * 3 + subcol + 1
                        if num in candidates:
                            line += f" {num} "
                        else:
                            line += "   "
                    line += " "
                
                if (col + 1) % 3 == 0:
                    line += "|"
            
            print(line)
        
        if (row + 1) % 3 == 0:
            print("+" + "-------------------+" * 3)


if __name__ == "__main__":
    #Test the Sudoku class
    print("Testing Sudoku class...")
    
    #Create example puzzle
    sudoku = create_example_sudoku()
    
    print("\nPuzzle loaded:")
    print(sudoku)
    
    print(f"\nDifficulty: {sudoku.difficulty}")
    print(f"Empty cells: {len(sudoku.get_empty_cells())}")
    print(f"Is solved: {sudoku.is_solved()}")
    print(f"Is valid: {sudoku.is_valid_solution()}")
    
    #Test candidates
    print("\nCandidates for first empty cell:")
    empty_cells = sudoku.get_empty_cells()
    if empty_cells:
        row, col = empty_cells[0]
        candidates = sudoku.get_candidates(row, col)
        print(f"Cell ({row}, {col}): {candidates}")
    
    #Test graph representation
    print("\nGraph representation:")
    graph_data = sudoku.to_graph_representation()
    print(f"Vertices: {len(graph_data['vertices'])}")
    print(f"Edges: {len(graph_data['edges'])}")
    print(f"Colored vertices: {len(graph_data['colors'])}")
    
    #Print graph statistics
    if sudoku.graph:
        stats = sudoku.graph.get_graph_statistics()
        print("\nGraph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")