# Maze Generator - Recursive Backtracker Algorithm

A Python implementation of a maze generator using the Recursive Backtracker (Depth-First Search) algorithm with real-time Pygame visualization.

## Features

- ✨ **Perfect Mazes**: Generates mazes with exactly one unique path between any two cells
- 🎮 **Real-time Visualization**: Watch the algorithm work with Pygame animation
- 🎲 **Reproducible**: Use seeds for deterministic maze generation
- 💾 **Save/Load**: Export and import mazes in NumPy format
- 🎯 **Start/Goal Generation**: Automatic start (top-left) and random goal positioning
- ✅ **Validation**: Built-in maze validation to ensure correctness

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install numpy matplotlib pygame --break-system-packages
```

## Usage

### Quick Start - Just Run It!

```bash
python maze_generator.py
```

This will:
1. Generate a maze with **random seed** (different every time!)
2. Show **real-time Pygame visualization**
3. Save three output files:
   - `maze_output.npy` - NumPy array format
   - `maze.txt` - Text format (# = wall, space = path)
   - `maze_output.png` - PNG visualization with start (white) and goal (red)

### Customize Maze Size

Edit the `main()` function in `maze_generator.py`:

```python
# CONFIGURATION - Edit these values!
width = 10   # Number of cells horizontally
height = 10  # Number of cells vertically
```

Then run:
```bash
python maze_generator.py
```

Every run generates a **completely new maze** with a random seed based on current time!

### Advanced Usage (Custom Seed)

If you need reproducible mazes for testing:

```python
from maze_generator import MazeGenerator

# Create a 15x15 maze with visualization
generator = MazeGenerator(width=15, height=15, seed=42)
maze = generator.generate(visualize=True, delay=0.02)

# Validate the maze
generator.validate_maze()

# Save the maze
generator.save_maze("my_maze.npy")
```

### Running the Demo

Simply run the main script:

```bash
python maze_generator.py
```

This will:
1. Generate a maze (random seed from current time)
2. Display real-time Pygame visualization
3. Validate the maze structure
4. Save outputs: `.npy`, `.txt`, and `.png` files

### Running Examples

```bash
python maze_examples.py
```

This runs multiple tests and demos:
- Reproducibility test
- Different maze sizes
- Save/load demonstration
- Matplotlib visualization

## Maze Representation

The maze uses an **expanded grid representation**: `(2*H+1) x (2*W+1)` boolean array where:
- `True` = Wall
- `False` = Path

This representation explicitly includes walls between cells, making it easy to:
- Visualize the maze
- Implement pathfinding algorithms
- Export to various formats

### Example: 3x3 Cell Maze

```
Cell space: 3 x 3
Grid space: 7 x 7 (2*3+1 x 2*3+1)
```

## API Reference

### MazeGenerator Class

#### Constructor

```python
MazeGenerator(width: int, height: int, seed: Optional[int] = None)
```

**Parameters:**
- `width`: Number of cells horizontally
- `height`: Number of cells vertically
- `seed`: Random seed for reproducibility (optional)

#### Methods

##### generate()

```python
generate(visualize: bool = False, delay: float = 0.01) -> np.ndarray
```

Generate the maze using recursive backtracker algorithm.

**Parameters:**
- `visualize`: Whether to show real-time Pygame visualization
- `delay`: Delay between steps in seconds (for visualization speed)

**Returns:**
- 2D boolean NumPy array representing the maze

##### get_start_position()

```python
get_start_position() -> Tuple[int, int]
```

Get the start position (always top-left corner in grid coordinates).

**Returns:**
- Tuple of (row, col) in grid coordinates

##### get_random_goal_position()

```python
get_random_goal_position(seed: Optional[int] = None) -> Tuple[int, int]
```

Get a random goal position that's not the start.

**Parameters:**
- `seed`: Optional seed for goal randomization

**Returns:**
- Tuple of (row, col) in grid coordinates

##### save_maze()

```python
save_maze(filename: str)
```

Save the maze to a NumPy file.

##### load_maze()

```python
load_maze(filename: str)
```

Load a maze from a NumPy file.

##### validate_maze()

```python
validate_maze() -> bool
```

Validate that the maze is perfect (follows tree property).

**Returns:**
- `True` if maze is valid, `False` otherwise

## Visualization

### Pygame Real-time Visualization

The Pygame visualization shows:
- **Black squares**: Unvisited cells
- **Dark green squares**: Explored paths and removed walls
- **Bright green lines**: Walls (1/5 width of cells)
- **White square**: Current position of the algorithm (fills entire cell)

After generation completes, the window stays open so you can view the final maze. Close the window manually to continue.

### Matplotlib Static Visualization

Use the utility function in `maze_examples.py`:

```python
from maze_examples import visualize_with_matplotlib

visualize_with_matplotlib(maze, start=(1,1), goal=(29,29), 
                         save_path="maze.png")
```

## Algorithm: Recursive Backtracker

The Recursive Backtracker (Depth-First Search) algorithm:

1. Start at the top-left cell
2. Mark current cell as visited
3. While there are unvisited cells:
   - If current cell has unvisited neighbors:
     - Choose random unvisited neighbor
     - Remove wall between current and neighbor
     - Move to neighbor and mark as visited
   - Else (no unvisited neighbors):
     - Backtrack to previous cell
4. Continue until all cells are visited

### Why Recursive Backtracker?

- ✅ Simple to implement
- ✅ Fast execution
- ✅ Produces visually pleasing mazes with long corridors
- ✅ Guarantees perfect maze (one unique path between any two points)
- ✅ Low memory footprint

## Testing & Validation

### Validation Criteria

A perfect maze must satisfy:
- All cells are visited
- Number of removed walls = (Width × Height - 1)
- No isolated sections
- Exactly one path between any two cells

### Running Tests

```python
# Test reproducibility
from maze_examples import test_reproducibility
test_reproducibility()

# Test different sizes
from maze_examples import test_different_sizes
test_different_sizes()
```

## Integration with RL Agent

Export maze for reinforcement learning:

```python
from maze_examples import export_for_rl_agent

maze_data = export_for_rl_agent(width=20, height=20, seed=42)

# maze_data contains:
# - grid: 2D boolean array
# - width, height: dimensions
# - start: start position
# - goal: goal position
# - seed: random seed used
```

## File Structure

```
.
├── maze_generator.py    # Main maze generator class
├── maze_examples.py     # Examples and utilities
├── README.md           # This file
└── maze_output.npy     # Generated maze (after running)
```

## Customization

### Adjusting Visualization Speed

```python
# Slower visualization
maze = generator.generate(visualize=True, delay=0.1)

# Faster visualization
maze = generator.generate(visualize=True, delay=0.001)
```

### Changing Wall Thickness

Edit the `cell_size` and `wall_thickness` variables in the `generate()` method:

```python
cell_size = 30
wall_thickness = cell_size // 5  # Walls are 1/5 of cell size
```

### Changing Colors

Modify the color constants in the `generate()` method:

```python
BLACK = (0, 0, 0)           # Unvisited cells
DARK_GREEN = (0, 100, 0)    # Explored path
BRIGHT_GREEN = (0, 255, 0)  # Walls
WHITE = (255, 255, 255)     # Current position
```

## Performance

| Maze Size | Generation Time | Validation |
|-----------|----------------|------------|
| 5x5       | < 0.1s        | ✓          |
| 10x10     | < 0.2s        | ✓          |
| 25x25     | < 1s          | ✓          |
| 50x50     | < 3s          | ✓          |

*Times are approximate and exclude visualization delay*

## References

- [Maze Generation Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Maze_generation_algorithm)
- Recursive Backtracker is also known as "Depth-First Search Maze Generation"
- Based on the project documentation provided in `maze_generator_instructions.txt`

## License

This code is provided for educational purposes as part of the Autonomous Maze Solver project.

## Acknowledgments

- Algorithm based on the classic Recursive Backtracker approach
- Visualization inspired by common maze generation tutorials
- Implementation follows the specifications in the project documentation