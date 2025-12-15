"""
Sample Sudoku puzzles for testing and demonstration
"""

from typing import Dict, List
import json


# Collection of Sudoku puzzles with different difficulties
PUZZLES: Dict[str, Dict] = {
    "very_easy": {
        "name": "Very Easy Puzzle",
        "difficulty": "Very Easy",
        "description": "Almost solved puzzle for testing",
        "puzzle": [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
    },
    
    "easy": {
        "name": "Easy Puzzle",
        "difficulty": "Easy",
        "description": "Standard easy difficulty puzzle",
        "puzzle": [
            [0, 0, 0, 2, 6, 0, 7, 0, 1],
            [6, 8, 0, 0, 7, 0, 0, 9, 0],
            [1, 9, 0, 0, 0, 4, 5, 0, 0],
            [8, 2, 0, 1, 0, 0, 0, 4, 0],
            [0, 0, 4, 6, 0, 2, 9, 0, 0],
            [0, 5, 0, 0, 0, 3, 0, 2, 8],
            [0, 0, 9, 3, 0, 0, 0, 7, 4],
            [0, 4, 0, 0, 5, 0, 0, 3, 6],
            [7, 0, 3, 0, 1, 8, 0, 0, 0]
        ]
    },
    
    "medium": {
        "name": "Medium Puzzle",
        "difficulty": "Medium",
        "description": "Standard medium difficulty puzzle",
        "puzzle": [
            [0, 2, 0, 6, 0, 8, 0, 0, 0],
            [5, 8, 0, 0, 0, 9, 7, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0],
            [3, 7, 0, 0, 0, 0, 5, 0, 0],
            [6, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 8, 0, 0, 0, 0, 1, 3],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 9, 8, 0, 0, 0, 3, 6],
            [0, 0, 0, 3, 0, 6, 0, 9, 0]
        ]
    },
    
    "hard": {
        "name": "Hard Puzzle",
        "difficulty": "Hard",
        "description": "Challenging hard difficulty puzzle",
        "puzzle": [
            [0, 0, 0, 6, 0, 0, 4, 0, 0],
            [7, 0, 0, 0, 0, 3, 6, 0, 0],
            [0, 0, 0, 0, 9, 1, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 1, 8, 0, 0, 0, 3],
            [0, 0, 0, 3, 0, 6, 0, 4, 5],
            [0, 4, 0, 2, 0, 0, 0, 6, 0],
            [9, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 1, 0, 0]
        ]
    },
    
    "expert": {
        "name": "Expert Puzzle",
        "difficulty": "Expert",
        "description": "Very difficult expert puzzle",
        "puzzle": [
            [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0, 3],
            [0, 7, 4, 0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0, 2],
            [0, 8, 0, 0, 4, 0, 0, 1, 0],
            [6, 0, 0, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 7, 8, 0],
            [5, 0, 0, 0, 0, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0]
        ]
    },
    
    "worlds_hardest": {
        "name": "World's Hardest Sudoku",
        "difficulty": "Extreme",
        "description": "Reported as one of the world's hardest Sudoku puzzles",
        "puzzle": [
            [8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 6, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 9, 0, 2, 0, 0],
            [0, 5, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 4, 5, 7, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 3, 0],
            [0, 0, 1, 0, 0, 0, 0, 6, 8],
            [0, 0, 8, 5, 0, 0, 0, 1, 0],
            [0, 9, 0, 0, 0, 0, 4, 0, 0]
        ]
    },
    
    "empty": {
        "name": "Empty Grid",
        "difficulty": "N/A",
        "description": "Completely empty Sudoku grid",
        "puzzle": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    },
    
    "solved": {
        "name": "Solved Puzzle",
        "difficulty": "Solved",
        "description": "Completely solved Sudoku puzzle",
        "puzzle": [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
    }
}


# String representations of puzzles (81 characters)
PUZZLE_STRINGS: Dict[str, str] = {
    "very_easy": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "easy": "000260701680700090190004500820100040004602900050003028009300074040050036703018000",
    "medium": "020608000580097000000040000370000500600000004008000013000020000009800306000306090",
    "hard": "000600400700003600000910080000000000050180003000306045040200060903000000020000100",
    "expert": "020000000000600003074080000000003002080040010600500000000010780500009000000000040",
    "worlds_hardest": "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "empty": "0" * 81,
    "solved": "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
}


def get_puzzle_by_name(name: str) -> Dict:
    """
    Get puzzle data by name
    
    Args:
        name: Puzzle name (e.g., "easy", "medium", "hard")
        
    Returns:
        Dictionary with puzzle data
    """
    if name in PUZZLES:
        return PUZZLES[name]
    else:
        raise ValueError(f"Unknown puzzle name: {name}. Available: {list(PUZZLES.keys())}")


def get_puzzle_string_by_name(name: str) -> str:
    """
    Get puzzle string by name
    
    Args:
        name: Puzzle name
        
    Returns:
        81-character puzzle string
    """
    if name in PUZZLE_STRINGS:
        return PUZZLE_STRINGS[name]
    else:
        raise ValueError(f"Unknown puzzle name: {name}. Available: {list(PUZZLE_STRINGS.keys())}")


def list_all_puzzles() -> List[str]:
    """
    List all available puzzle names
    
    Returns:
        List of puzzle names
    """
    return list(PUZZLES.keys())


def print_puzzle_info(name: str):
    """
    Print information about a puzzle
    
    Args:
        name: Puzzle name
    """
    if name not in PUZZLES:
        print(f"Unknown puzzle: {name}")
        return
    
    puzzle_data = PUZZLES[name]
    
    print(f"\n{'='*60}")
    print(f"PUZZLE: {puzzle_data['name']}")
    print(f"{'='*60}")
    print(f"Difficulty: {puzzle_data['difficulty']}")
    print(f"Description: {puzzle_data['description']}")
    
    # Count empty cells
    empty_count = sum(1 for row in puzzle_data['puzzle'] for cell in row if cell == 0)
    print(f"Empty cells: {empty_count}/81")
    
    # Print the puzzle grid
    print("\nGrid:")
    print("+" + "---+" * 3)
    for i, row in enumerate(puzzle_data['puzzle']):
        line = "|"
        for j, cell in enumerate(row):
            if cell == 0:
                line += " . "
            else:
                line += f" {cell} "
            
            if (j + 1) % 3 == 0:
                line += "|"
        print(line)
        
        if (i + 1) % 3 == 0:
            print("+" + "---+" * 3)


def save_puzzles_to_file(filename: str = "all_puzzles.json"):
    """
    Save all puzzles to a JSON file
    
    Args:
        filename: Output filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(PUZZLES, f, indent=2)
        print(f"All puzzles saved to {filename}")
    except Exception as e:
        print(f"Error saving puzzles: {e}")


def load_puzzles_from_file(filename: str = "all_puzzles.json") -> Dict:
    """
    Load puzzles from a JSON file
    
    Args:
        filename: Input filename
        
    Returns:
        Dictionary of puzzles
    """
    try:
        with open(filename, 'r') as f:
            puzzles = json.load(f)
        print(f"Loaded puzzles from {filename}")
        return puzzles
    except Exception as e:
        print(f"Error loading puzzles: {e}")
        return {}


def create_puzzle_from_grid(grid: List[List[int]], name: str = "custom") -> Dict:
    """
    Create a puzzle dictionary from a grid
    
    Args:
        grid: 9x9 grid of integers (0 for empty)
        name: Puzzle name
        
    Returns:
        Puzzle dictionary
    """
    # Validate grid
    if len(grid) != 9:
        raise ValueError("Grid must have 9 rows")
    
    for row in grid:
        if len(row) != 9:
            raise ValueError("Each row must have 9 columns")
        
        for cell in row:
            if not 0 <= cell <= 9:
                raise ValueError(f"Invalid cell value: {cell}. Must be 0-9.")
    
    # Count empty cells to estimate difficulty
    empty_count = sum(1 for row in grid for cell in row if cell == 0)
    
    if empty_count <= 30:
        difficulty = "Easy"
    elif empty_count <= 45:
        difficulty = "Medium"
    elif empty_count <= 60:
        difficulty = "Hard"
    else:
        difficulty = "Expert"
    
    return {
        "name": name,
        "difficulty": difficulty,
        "description": f"Custom puzzle with {empty_count} empty cells",
        "puzzle": grid
    }


def generate_puzzle_collection(count: int = 10) -> List[Dict]:
    """
    Generate a collection of random puzzles
    
    Args:
        count: Number of puzzles to generate
        
    Returns:
        List of puzzle dictionaries
    """
    from src.utils import generate_random_sudoku
    
    puzzles = []
    
    difficulties = ["easy", "medium", "hard", "expert"]
    
    for i in range(count):
        # Distribute difficulties
        difficulty = difficulties[i % len(difficulties)]
        
        # Generate puzzle
        sudoku = generate_random_sudoku(difficulty)
        
        # Create puzzle dictionary
        puzzle_dict = {
            "name": f"Random {difficulty.capitalize()} {i+1}",
            "difficulty": difficulty.capitalize(),
            "description": f"Randomly generated {difficulty} puzzle",
            "puzzle": sudoku.grid
        }
        
        puzzles.append(puzzle_dict)
    
    return puzzles


if __name__ == "__main__":
    # Test the puzzles module
    print("Testing Sudoku Puzzles Module...")
    
    # List all puzzles
    print(f"\nAvailable puzzles: {list_all_puzzles()}")
    
    # Print info for each puzzle
    for puzzle_name in ["very_easy", "medium", "worlds_hardest"]:
        print_puzzle_info(puzzle_name)
    
    # Test getting puzzle data
    print("\nTesting puzzle retrieval:")
    puzzle_data = get_puzzle_by_name("easy")
    print(f"Retrieved 'easy' puzzle: {puzzle_data['name']}")
    
    # Test getting puzzle string
    puzzle_str = get_puzzle_string_by_name("medium")
    print(f"\nMedium puzzle string (first 20 chars): {puzzle_str[:20]}...")
    
    # Test creating custom puzzle
    print("\nTesting custom puzzle creation:")
    custom_grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    custom_puzzle = create_puzzle_from_grid(custom_grid, "my_puzzle")
    print(f"Created custom puzzle: {custom_puzzle['name']}")
    print(f"Difficulty: {custom_puzzle['difficulty']}")
    
    # Save puzzles to file
    print("\nSaving puzzles to file...")
    save_puzzles_to_file("test_puzzles.json")
    
    print("\nPuzzles module test complete!")