"""
Visualization module for Sudoku Graph Coloring
Uses matplotlib for static visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import os

from .sudoku import Sudoku
from .solver import SudokuSolver


class SudokuVisualizer:
    """Visualizes Sudoku solving process using graph coloring"""
    
    def __init__(self, sudoku: Sudoku = None):
        """
        Initialize visualizer
        
        Args:
            sudoku: Optional Sudoku puzzle to visualize
        """
        self.sudoku = sudoku
        self.fig = None
        self.ax = None
        self.ax_graph = None
        self.animation = None
        self.colors_map = {
            1: '#FF6B6B',    # Red
            2: '#4ECDC4',    # Teal
            3: '#FFD166',    # Yellow
            4: '#06D6A0',    # Green
            5: '#118AB2',    # Blue
            6: '#073B4C',    # Dark Blue
            7: '#EF476F',    # Pink
            8: '#7B5E7B',    # Purple
            9: '#FF9A76',    # Orange
            0: '#FFFFFF'     # White (empty)
        }
        
        # For animation
        self.history_index = 0
        self.solver_history = []
        
    def set_sudoku(self, sudoku: Sudoku):
        """Set Sudoku puzzle to visualize"""
        self.sudoku = sudoku
    
    def plot_sudoku_grid(self, title: str = "Sudoku Puzzle", 
                        show_candidates: bool = False,
                        highlight_conflicts: bool = False):
        """
        Plot Sudoku grid
        
        Args:
            title: Plot title
            show_candidates: Whether to show candidate numbers
            highlight_conflicts: Whether to highlight conflicts
        """
        if self.sudoku is None:
            print("Error: No Sudoku puzzle set")
            return
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, 9)
        self.ax.set_ylim(0, 9)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # So row 0 is at top
        
        # Hide axes
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Draw thick lines for 3x3 boxes
        for i in range(0, 10, 3):
            self.ax.axhline(i, color='black', linewidth=3)
            self.ax.axvline(i, color='black', linewidth=3)
        
        # Draw thin lines for cells
        for i in range(10):
            self.ax.axhline(i, color='black', linewidth=1, alpha=0.5)
            self.ax.axvline(i, color='black', linewidth=1, alpha=0.5)
        
        # Fill cells with numbers
        for row in range(9):
            for col in range(9):
                value = self.sudoku.get_cell(row, col)
                
                # Determine cell color
                cell_color = self.colors_map.get(value, '#FFFFFF')
                
                # Highlight initial cells
                if (self.sudoku.initial_state and 
                    self.sudoku.initial_state[row][col] != 0):
                    # Initial given cells get darker background
                    rect = patches.Rectangle(
                        (col, row), 1, 1,
                        linewidth=0,
                        facecolor='#F0F0F0',
                        alpha=0.7
                    )
                    self.ax.add_patch(rect)
                
                # Draw cell
                rect = patches.Rectangle(
                    (col, row), 1, 1,
                    linewidth=1,
                    edgecolor='black',
                    facecolor=cell_color,
                    alpha=0.3
                )
                self.ax.add_patch(rect)
                
                # Add number if not zero
                if value != 0:
                    # Determine text color based on background
                    text_color = 'black' if value <= 5 else 'darkblue'
                    
                    # Bold for initial values
                    weight = 'bold' if (self.sudoku.initial_state and 
                                      self.sudoku.initial_state[row][col] != 0) else 'normal'
                    
                    self.ax.text(
                        col + 0.5, row + 0.5,
                        str(value),
                        fontsize=20,
                        fontweight=weight,
                        color=text_color,
                        ha='center',
                        va='center'
                    )
                
                # Show candidates if requested
                elif show_candidates:
                    candidates = self.sudoku.get_candidates(row, col)
                    if candidates:
                        # Create 3x3 grid of candidate numbers
                        for num in range(1, 10):
                            if num in candidates:
                                cand_row = (num - 1) // 3
                                cand_col = (num - 1) % 3
                                x = col + cand_col * 0.33 + 0.17
                                y = row + cand_row * 0.33 + 0.17
                                
                                self.ax.text(
                                    x, y,
                                    str(num),
                                    fontsize=8,
                                    color='gray',
                                    ha='center',
                                    va='center',
                                    alpha=0.7
                                )
        
        # Highlight conflicts if requested
        if highlight_conflicts and self.sudoku.graph:
            conflicts = self.sudoku.graph.get_color_conflicts()
            for v1, v2 in conflicts:
                row1, col1 = divmod(v1, 9)
                row2, col2 = divmod(v2, 9)
                
                # Highlight conflicting cells in red
                for (r, c) in [(row1, col1), (row2, col2)]:
                    rect = patches.Rectangle(
                        (c, r), 1, 1,
                        linewidth=3,
                        edgecolor='red',
                        facecolor='none',
                        alpha=0.7
                    )
                    self.ax.add_patch(rect)
        
        # Add row and column labels
        for i in range(9):
            # Row labels (A-I)
            self.ax.text(
                -0.5, i + 0.5,
                chr(ord('A') + i),
                fontsize=12,
                ha='center',
                va='center',
                color='gray'
            )
            
            # Column labels (1-9)
            self.ax.text(
                i + 0.5, -0.5,
                str(i + 1),
                fontsize=12,
                ha='center',
                va='center',
                color='gray'
            )
        
        plt.tight_layout()
    
    def plot_graph_representation(self, title: str = "Sudoku Constraint Graph"):
        """
        Plot the graph representation of Sudoku
        
        Args:
            title: Plot title
        """
        if self.sudoku is None or self.sudoku.graph is None:
            print("Error: No Sudoku graph available")
            return
        
        graph = self.sudoku.graph
        graph_data = self.sudoku.to_graph_representation()
        
        self.fig, self.ax_graph = plt.subplots(figsize=(12, 10))
        
        # Set up plot
        self.ax_graph.set_xlim(-1, 10)
        self.ax_graph.set_ylim(-1, 10)
        self.ax_graph.set_aspect('equal')
        self.ax_graph.set_title(title, fontsize=16, fontweight='bold')
        self.ax_graph.set_xticks([])
        self.ax_graph.set_yticks([])
        
        # Draw edges (with reduced opacity to avoid clutter)
        for edge in graph_data['edges'][::10]:  # Sample edges to avoid clutter
            v1 = edge['from']
            v2 = edge['to']
            
            x1, y1 = graph.vertex_positions.get(v1, (0, 0))
            x2, y2 = graph.vertex_positions.get(v2, (0, 0))
            
            self.ax_graph.plot(
                [x1, x2], [y1, y2],
                color='gray',
                linewidth=0.5,
                alpha=0.2,
                zorder=1
            )
        
        # Draw vertices
        for vertex in graph_data['vertices']:
            v = vertex['id']
            x, y = vertex['x'], vertex['y']
            color_value = graph.colors.get(v)
            
            # Vertex color based on assigned value
            if color_value is not None:
                face_color = self.colors_map.get(color_value, '#FFFFFF')
                edge_color = 'darkred' if (self.sudoku.initial_state and 
                                          self.sudoku.initial_state[vertex['row']][vertex['col']] != 0) else 'black'
                line_width = 3 if (self.sudoku.initial_state and 
                                 self.sudoku.initial_state[vertex['row']][vertex['col']] != 0) else 1
            else:
                face_color = '#FFFFFF'
                edge_color = 'gray'
                line_width = 1
            
            # Draw vertex circle
            circle = patches.Circle(
                (x, y),
                radius=0.3,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=0.8,
                zorder=2
            )
            self.ax_graph.add_patch(circle)
            
            # Add vertex label (value or vertex number)
            label = str(color_value) if color_value is not None else ""
            self.ax_graph.text(
                x, y,
                label,
                fontsize=10,
                ha='center',
                va='center',
                fontweight='bold',
                color='black' if color_value is not None else 'gray',
                zorder=3
            )
            
            # Add small coordinate label
            coord_label = f"{vertex['row']+1},{vertex['col']+1}"
            self.ax_graph.text(
                x, y - 0.5,
                coord_label,
                fontsize=6,
                ha='center',
                va='center',
                color='darkgray',
                alpha=0.7
            )
        
        # Add legend for colors
        legend_x = 8.5
        legend_y = 8.5
        
        self.ax_graph.text(
            legend_x, legend_y + 0.5,
            "Color Legend:",
            fontsize=10,
            ha='center',
            fontweight='bold'
        )
        
        for i in range(1, 4):  # Show first 3 colors
            circle = patches.Circle(
                (legend_x - 0.3, legend_y - i * 0.4),
                radius=0.15,
                facecolor=self.colors_map.get(i, '#FFFFFF'),
                edgecolor='black',
                linewidth=1
            )
            self.ax_graph.add_patch(circle)
            self.ax_graph.text(
                legend_x + 0.2, legend_y - i * 0.4,
                f" = {i}",
                fontsize=9,
                ha='left',
                va='center'
            )
        
        plt.tight_layout()
    
    def plot_side_by_side(self, title: str = "Sudoku Graph Coloring"):
        """
        Plot Sudoku grid and graph side by side
        
        Args:
            title: Main title
        """
        if self.sudoku is None:
            print("Error: No Sudoku puzzle set")
            return
        
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot Sudoku grid on left
        self.ax = ax1
        self.plot_sudoku_grid(title="Sudoku Grid", show_candidates=False)
        
        # Plot graph on right
        self.ax_graph = ax2
        self.plot_graph_representation(title="Constraint Graph")
        
        # Main title
        self.fig.suptitle(title, fontsize=20, fontweight='bold')
        plt.tight_layout()
    
    def animate_solving(self, solver: SudokuSolver, 
                       interval: int = 100,
                       save_path: Optional[str] = None):
        """
        Animate the solving process
        
        Args:
            solver: SudokuSolver with solution history
            interval: Animation interval in milliseconds
            save_path: Optional path to save animation as GIF
        """
        if not solver.solution_history:
            print("Error: No solving history available")
            return
        
        self.solver_history = solver.solution_history
        self.history_index = 0
        
        # Create figure
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Initial plot
        def init():
            ax1.clear()
            ax2.clear()
            return ax1, ax2
        
        # Animation update function
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # Get state at this frame
            state = self.solver_history[frame]
            grid = state['grid']
            description = state['description']
            
            # Update Sudoku grid
            ax1.set_xlim(0, 9)
            ax1.set_ylim(0, 9)
            ax1.set_aspect('equal')
            ax1.invert_yaxis()
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Draw grid lines
            for i in range(0, 10, 3):
                ax1.axhline(i, color='black', linewidth=3)
                ax1.axvline(i, color='black', linewidth=3)
            
            for i in range(10):
                ax1.axhline(i, color='black', linewidth=1, alpha=0.5)
                ax1.axvline(i, color='black', linewidth=1, alpha=0.5)
            
            # Fill cells
            for row in range(9):
                for col in range(9):
                    value = grid[row][col]
                    
                    # Cell color
                    cell_color = self.colors_map.get(value, '#FFFFFF')
                    
                    # Draw cell
                    rect = patches.Rectangle(
                        (col, row), 1, 1,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=cell_color,
                        alpha=0.3
                    )
                    ax1.add_patch(rect)
                    
                    # Add number if not zero
                    if value != 0:
                        text_color = 'black' if value <= 5 else 'darkblue'
                        ax1.text(
                            col + 0.5, row + 0.5,
                            str(value),
                            fontsize=20,
                            color=text_color,
                            ha='center',
                            va='center'
                        )
            
            ax1.set_title(f"Step {state['step']}: {description}", fontsize=14)
            
            # Update progress info
            info_text = (
                f"Steps: {state['step']}\n"
                f"Backtracks: {state.get('backtracks', 0)}\n"
                f"Colored: {state['graph_stats'].get('colored_vertices', 0)}/81\n"
                f"Conflicts: {state['graph_stats'].get('conflicts', 0)}"
            )
            
            ax1.text(
                0.02, 0.98,
                info_text,
                transform=ax1.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
            
            # Draw graph representation on right
            if self.sudoku and self.sudoku.graph:
                graph = self.sudoku.graph
                
                # Temporarily update graph colors
                original_colors = graph.colors.copy()
                
                # Update graph colors from grid
                for v in range(81):
                    row, col = divmod(v, 9)
                    graph.colors[v] = grid[row][col] if grid[row][col] != 0 else None
                
                # Draw graph
                ax2.set_xlim(-1, 10)
                ax2.set_ylim(-1, 10)
                ax2.set_aspect('equal')
                ax2.set_xticks([])
                ax2.set_yticks([])
                
                # Sample some edges
                edges_to_draw = []
                for u in range(81):
                    for v in graph.adjacency_list[u]:
                        if u < v and (u + v) % 20 == 0:  # Sample edges
                            edges_to_draw.append((u, v))
                
                for u, v in edges_to_draw:
                    x1, y1 = graph.vertex_positions.get(u, (0, 0))
                    x2, y2 = graph.vertex_positions.get(v, (0, 0))
                    
                    ax2.plot(
                        [x1, x2], [y1, y2],
                        color='gray',
                        linewidth=0.5,
                        alpha=0.2
                    )
                
                # Draw vertices
                for v in range(81):
                    if v in graph.vertex_positions:
                        x, y = graph.vertex_positions[v]
                        color_value = graph.colors[v]
                        
                        if color_value is not None:
                            face_color = self.colors_map.get(color_value, '#FFFFFF')
                            edge_color = 'black'
                        else:
                            face_color = '#FFFFFF'
                            edge_color = 'gray'
                        
                        circle = patches.Circle(
                            (x, y),
                            radius=0.3,
                            facecolor=face_color,
                            edgecolor=edge_color,
                            linewidth=1,
                            alpha=0.8
                        )
                        ax2.add_patch(circle)
                        
                        if color_value is not None:
                            ax2.text(
                                x, y,
                                str(color_value),
                                fontsize=10,
                                ha='center',
                                va='center',
                                fontweight='bold'
                            )
                
                ax2.set_title("Constraint Graph", fontsize=14)
                
                # Restore original colors
                graph.colors = original_colors
            
            return ax1, ax2
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            update,
            frames=len(self.solver_history),
            init_func=init,
            interval=interval,
            blit=False
        )
        
        # Save animation if requested
        if save_path:
            try:
                self.animation.save(save_path, writer='pillow', fps=10)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_algorithm_comparison(self, comparison_results: Dict[str, Dict]):
        """
        Plot comparison of different algorithms
        
        Args:
            comparison_results: Results from solver.compare_algorithms()
        """
        if not comparison_results:
            print("Error: No comparison results")
            return
        
        algorithms = list(comparison_results.keys())
        steps = [results['steps'] for results in comparison_results.values()]
        times = [results['time'] for results in comparison_results.values()]
        backtracks = [results.get('backtracks', 0) for results in comparison_results.values()]
        success = [results['success'] for results in comparison_results.values()]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Steps comparison
        bars1 = axes[0, 0].bar(algorithms, steps, color='skyblue')
        axes[0, 0].set_title('Number of Steps', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Steps')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')
        
        # Plot 2: Time comparison
        bars2 = axes[0, 1].bar(algorithms, times, color='lightcoral')
        axes[0, 1].set_title('Solving Time (seconds)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom')
        
        # Plot 3: Backtracks comparison
        bars3 = axes[1, 0].bar(algorithms, backtracks, color='lightgreen')
        axes[1, 0].set_title('Number of Backtracks', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Backtracks')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')
        
        # Plot 4: Success rate
        success_colors = ['green' if s else 'red' for s in success]
        bars4 = axes[1, 1].bar(algorithms, [1 if s else 0 for s in success], 
                              color=success_colors)
        axes[1, 1].set_title('Success (1 = Yes, 0 = No)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Success')
        axes[1, 1].set_ylim(0, 1.2)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, s in zip(bars4, success):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., 0.5,
                           'SUCCESS' if s else 'FAILED',
                           ha='center', va='center',
                           color='white', fontweight='bold')
        
        plt.suptitle('Algorithm Comparison', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_solving_progress(self, solver: SudokuSolver):
        """
        Plot solving progress over time
        
        Args:
            solver: SudokuSolver with history
        """
        if not solver.solution_history:
            print("Error: No solving history")
            return
        
        steps = [state['step'] for state in solver.solution_history]
        colored_counts = [state['graph_stats'].get('colored_vertices', 0) 
                         for state in solver.solution_history]
        conflicts = [state['graph_stats'].get('conflicts', 0) 
                    for state in solver.solution_history]
        backtracks = [state.get('backtracks', 0) 
                     for state in solver.solution_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Colored vertices over steps
        axes[0, 0].plot(steps, colored_counts, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Colored Vertices')
        axes[0, 0].set_title('Progress: Colored Vertices', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 81)
        
        # Plot 2: Conflicts over steps
        axes[0, 1].plot(steps, conflicts, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Conflicts')
        axes[0, 1].set_title('Conflicts Over Time', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Backtracks over steps
        axes[1, 0].plot(steps, backtracks, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Backtracks')
        axes[1, 0].set_title('Backtracks Over Time', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Step efficiency (colored per step)
        if len(steps) > 1:
            efficiency = []
            for i in range(1, len(steps)):
                if steps[i] != steps[i-1]:
                    eff = (colored_counts[i] - colored_counts[i-1]) / (steps[i] - steps[i-1])
                    efficiency.append(eff)
            
            axes[1, 1].plot(range(len(efficiency)), efficiency, 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Interval')
            axes[1, 1].set_ylabel('Vertices Colored per Step')
            axes[1, 1].set_title('Solving Efficiency', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Solving Progress: {solver.algorithm_used}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def save_all_plots(self, sudoku: Sudoku, solver: SudokuSolver, 
                      output_dir: str = "sudoku_plots"):
        """
        Save all visualization plots to files
        
        Args:
            sudoku: Sudoku puzzle
            solver: SudokuSolver with results
            output_dir: Output directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self.set_sudoku(sudoku)
        
        # Save Sudoku grid
        self.plot_sudoku_grid("Sudoku Puzzle")
        plt.savefig(f"{output_dir}/sudoku_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save graph representation
        self.plot_graph_representation("Constraint Graph")
        plt.savefig(f"{output_dir}/constraint_graph.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save side-by-side
        self.plot_side_by_side("Sudoku Graph Coloring")
        plt.savefig(f"{output_dir}/side_by_side.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save solved puzzle if solved
        if sudoku.is_solved():
            self.plot_sudoku_grid("Solved Sudoku")
            plt.savefig(f"{output_dir}/solved_sudoku.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {output_dir}/")
    
    def show(self):
        """Show current plot"""
        if self.fig is not None:
            plt.show()
        else:
            print("No plot to show. Call a plotting method first.")


def create_interactive_demo():
    """Create an interactive demo using matplotlib widgets"""
    print("Interactive demo would require matplotlib widgets.")
    print("This is a placeholder for interactive features.")
    print("For full interactivity, consider using a web-based visualization.")


if __name__ == "__main__":
    # Test the visualizer
    print("Testing Sudoku Visualizer...")
    
    from .sudoku import create_example_sudoku
    from .solver import SudokuSolver
    
    # Create and solve a puzzle
    sudoku = create_example_sudoku()
    solver = SudokuSolver(sudoku)
    
    print("\nOriginal Puzzle:")
    print(sudoku)
    
    # Solve quickly (just for visualization test)
    success = solver.solve_backtracking(use_heuristics=True)
    
    if success:
        print("\nSolved Puzzle:")
        print(sudoku)
    
    # Create visualizer
    visualizer = SudokuVisualizer(sudoku)
    
    # Plot various visualizations
    print("\nCreating visualizations...")
    
    # Plot 1: Sudoku grid
    visualizer.plot_sudoku_grid("Example Sudoku Puzzle")
    plt.show()
    
    # Plot 2: Graph representation
    visualizer.plot_graph_representation()
    plt.show()
    
    # Plot 3: Side by side
    visualizer.plot_side_by_side()
    plt.show()
    
    # Plot 4: Solving progress
    visualizer.plot_solving_progress(solver)
    plt.show()
    
    print("Visualization test complete!")