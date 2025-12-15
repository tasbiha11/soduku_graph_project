from typing import List, Dict, Set, Tuple, Optional
import random


class Graph:
    """Graph representation for Sudoku constraints"""
    
    def __init__(self, n_vertices: int = 81):
        """
        Initialize graph for Sudoku (81 vertices for 9x9 grid)
        
        Args:
            n_vertices: Number of vertices (81 for standard Sudoku)
        """
        self.n_vertices = n_vertices
        self.vertices = list(range(n_vertices))
        self.adjacency_list: Dict[int, Set[int]] = {v: set() for v in self.vertices}
        self.colors: Dict[int, Optional[int]] = {v: None for v in self.vertices}
        self.available_colors = list(range(1, 10))  #Colors 1-9 for Sudoku
        
        #Vertex properties for visualization
        self.vertex_positions: Dict[int, Tuple[float, float]] = {}
        self.vertex_labels: Dict[int, str] = {}
        
    def add_edge(self, u: int, v: int):
        """Add undirected edge between vertices u and v"""
        if u != v:  #No self-loops
            self.adjacency_list[u].add(v)
            self.adjacency_list[v].add(u)
    
    def add_edges_from_list(self, edges: List[Tuple[int, int]]):
        """Add multiple edges from list of tuples"""
        for u, v in edges:
            self.add_edge(u, v)
    
    def get_neighbors(self, v: int) -> Set[int]:
        """Get all neighbors of vertex v"""
        return self.adjacency_list[v]
    
    def degree(self, v: int) -> int:
        """Get degree (number of neighbors) of vertex v"""
        return len(self.adjacency_list[v])
    
    def get_uncolored_vertices(self) -> List[int]:
        """Get list of vertices that haven't been colored yet"""
        return [v for v in self.vertices if self.colors[v] is None]
    
    def get_colored_vertices(self) -> List[int]:
        """Get list of vertices that have been colored"""
        return [v for v in self.vertices if self.colors[v] is not None]
    
    def is_valid_color(self, v: int, color: int) -> bool:
        """
        Check if assigning color to vertex v is valid
        (no adjacent vertex has the same color)
        """
        for neighbor in self.adjacency_list[v]:
            if self.colors[neighbor] == color:
                return False
        return True
    
    def get_available_colors(self, v: int) -> List[int]:
        """
        Get list of colors that can be assigned to vertex v
        without violating constraints
        """
        used_colors = set()
        for neighbor in self.adjacency_list[v]:
            if self.colors[neighbor] is not None:
                used_colors.add(self.colors[neighbor])
        return [c for c in self.available_colors if c not in used_colors]
    
    def assign_color(self, v: int, color: int) -> bool:
        """
        Assign color to vertex v if valid
        
        Args:
            v: Vertex index
            color: Color to assign (1-9 for Sudoku)
            
        Returns:
            bool: True if assignment was successful, False if invalid
        """
        if self.is_valid_color(v, color):
            self.colors[v] = color
            return True
        return False
    
    def remove_color(self, v: int):
        """Remove color assignment from vertex v"""
        self.colors[v] = None
    
    def is_completely_colored(self) -> bool:
        """Check if all vertices have been colored"""
        return all(color is not None for color in self.colors.values())
    
    def is_valid_coloring(self) -> bool:
        """Verify if current coloring is valid (no adjacent vertices share color)"""
        for v in self.vertices:
            if self.colors[v] is not None:
                for neighbor in self.adjacency_list[v]:
                    if self.colors[neighbor] == self.colors[v]:
                        return False
        return True
    
    def get_color_conflicts(self) -> List[Tuple[int, int]]:
        """
        Get list of conflicting edges (adjacent vertices with same color)
        Returns list of (vertex1, vertex2) pairs
        """
        conflicts = []
        for v in self.vertices:
            if self.colors[v] is not None:
                for neighbor in self.adjacency_list[v]:
                    if (neighbor > v and  # Avoid duplicate pairs
                        self.colors[neighbor] == self.colors[v]):
                        conflicts.append((v, neighbor))
        return conflicts
    
    def reset_colors(self):
        """Reset all color assignments"""
        self.colors = {v: None for v in self.vertices}
    
    def set_vertex_position(self, v: int, x: float, y: float):
        """Set position for vertex (for visualization)"""
        self.vertex_positions[v] = (x, y)
    
    def set_vertex_label(self, v: int, label: str):
        """Set label for vertex (for visualization)"""
        self.vertex_labels[v] = label
    
    def get_vertex_at_position(self, x: float, y: float, radius: float = 0.5) -> Optional[int]:
        """
        Find vertex at given position (for interactive visualization)
        
        Args:
            x, y: Position coordinates
            radius: Search radius
            
        Returns:
            Vertex index or None if no vertex at position
        """
        for v, (vx, vy) in self.vertex_positions.items():
            if abs(vx - x) <= radius and abs(vy - y) <= radius:
                return v
        return None
    
    def get_most_constrained_vertex(self, uncolored_vertices: List[int]) -> int:
        """
        Get the most constrained vertex (minimum remaining values heuristic)
        
        Args:
            uncolored_vertices: List of vertices to choose from
            
        Returns:
            Vertex with fewest available colors
        """
        if not uncolored_vertices:
            return -1
        
        #Count available colors for each vertex
        vertex_constraints = []
        for v in uncolored_vertices:
            available_colors = self.get_available_colors(v)
            vertex_constraints.append((v, len(available_colors)))
        
        #Return vertex with minimum available colors
        #If tie, choose one with maximum degree
        min_constraint = min(vertex_constraints, key=lambda x: x[1])[1]
        candidates = [v for v, c in vertex_constraints if c == min_constraint]
        
        if len(candidates) == 1:
            return candidates[0]
        else:
            #Tie-break: choose vertex with highest degree
            return max(candidates, key=lambda v: self.degree(v))
    
    def get_least_constraining_color(self, v: int, available_colors: List[int]) -> int:
        """
        Get the least constraining color (least constraining value heuristic)
        
        Args:
            v: Vertex to color
            available_colors: List of colors that can be assigned
            
        Returns:
            Color that leaves maximum flexibility for neighbors
        """
        if not available_colors:
            return -1
        
        color_impact = []
        for color in available_colors:
            #Count how many neighbors would lose this color as option
            impact = 0
            for neighbor in self.adjacency_list[v]:
                if self.colors[neighbor] is None and color in self.get_available_colors(neighbor):
                    impact += 1
            color_impact.append((color, impact))
        
        #Return color with minimum impact (least constraining)
        return min(color_impact, key=lambda x: x[1])[0]
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """Get statistics about the graph"""
        degrees = [self.degree(v) for v in self.vertices]
        colored_count = len(self.get_colored_vertices())
        
        return {
            "vertices": self.n_vertices,
            "edges": sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2,
            "average_degree": sum(degrees) / len(degrees),
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "colored_vertices": colored_count,
            "uncolored_vertices": self.n_vertices - colored_count,
            "is_valid": self.is_valid_coloring(),
            "conflicts": len(self.get_color_conflicts())
        }
    
    def __str__(self) -> str:
        """String representation of graph"""
        stats = self.get_graph_statistics()
        return (f"Graph with {stats['vertices']} vertices, {stats['edges']} edges\n"
                f"Colored: {stats['colored_vertices']}, "
                f"Valid: {stats['is_valid']}, "
                f"Conflicts: {stats['conflicts']}")


def create_sudoku_graph() -> Graph:
    """
    Create graph representing Sudoku constraints
    
    Returns:
        Graph with 81 vertices and edges representing:
        - Same row constraints
        - Same column constraints  
        - Same 3x3 box constraints
    """
    graph = Graph(81)
    
    #Add edges for Sudoku constraints
    edges = []
    
    for i in range(81):
        row_i, col_i = divmod(i, 9)
        
        #Same row constraints
        for col in range(9):
            if col != col_i:
                j = row_i * 9 + col
                edges.append((i, j))
        
        #Same column constraints
        for row in range(9):
            if row != row_i:
                j = row * 9 + col_i
                edges.append((i, j))
        
        #Same 3x3 box constraints
        box_row = row_i // 3
        box_col = col_i // 3
        
        for r in range(box_row * 3, (box_row + 1) * 3):
            for c in range(box_col * 3, (box_col + 1) * 3):
                if r != row_i or c != col_i:
                    j = r * 9 + c
                    edges.append((i, j))
    
    #Remove duplicate edges (undirected graph)
    unique_edges = set()
    for u, v in edges:
        if u < v:
            unique_edges.add((u, v))
        else:
            unique_edges.add((v, u))
    
    graph.add_edges_from_list(list(unique_edges))
    
    #Set vertex positions for 9x9 grid layout
    for i in range(81):
        row, col = divmod(i, 9)
        #Center positions in range [0, 8]
        x = col
        y = 8 - row  # Flip y so row 0 is at top
        graph.set_vertex_position(i, x, y)
        
        #Set labels as cell coordinates (e.g., "A1", "B2")
        row_label = chr(ord('A') + row)
        col_label = str(col + 1)
        graph.set_vertex_label(i, f"{row_label}{col_label}")
    
    return graph


if __name__ == "__main__":
    #Test the graph creation
    print("Testing Sudoku Graph Creation...")
    graph = create_sudoku_graph()
    
    print(f"Created graph with {graph.n_vertices} vertices")
    
    #Test a few vertices
    test_vertices = [0, 40, 80]  # Top-left, center, bottom-right
    for v in test_vertices:
        row, col = divmod(v, 9)
        print(f"\nVertex {v} (Row {row}, Col {col}):")
        print(f"  Degree: {graph.degree(v)}")
        print(f"  Label: {graph.vertex_labels.get(v, 'N/A')}")
        print(f"  Position: {graph.vertex_positions.get(v, 'N/A')}")
    
    #Print statistics
    stats = graph.get_graph_statistics()
    print(f"\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")