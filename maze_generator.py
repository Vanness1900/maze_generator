import numpy as np
import pygame
import random
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class MazeGenerator:
    """
    Maze Generator using Recursive Backtracker (DFS) algorithm.
    Produces perfect mazes with one unique path between any two cells.
    """
    
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        """
        Initialize the maze generator.
        
        Args:
            width: Number of cells horizontally
            height: Number of cells vertically
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Expanded grid: (2*H+1) x (2*W+1) where True=wall, False=path
        self.grid = np.ones((2 * height + 1, 2 * width + 1), dtype=bool)
        
        # Track visited cells for the algorithm
        self.visited = np.zeros((height, width), dtype=bool)
        
        # Stack for backtracking
        self.stack = []
        
        # Current cell for visualization
        self.current_cell = None
        
    def _cell_to_grid(self, row: int, col: int) -> Tuple[int, int]:
        """Convert cell coordinates to grid coordinates."""
        return (2 * row + 1, 2 * col + 1)
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all unvisited neighbors of a cell."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                # Check if unvisited
                if not self.visited[new_row, new_col]:
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _remove_wall(self, cell1: Tuple[int, int], cell2: Tuple[int, int]):
        """Remove the wall between two adjacent cells."""
        r1, c1 = cell1
        r2, c2 = cell2
        
        # Convert to grid coordinates
        gr1, gc1 = self._cell_to_grid(r1, c1)
        gr2, gc2 = self._cell_to_grid(r2, c2)
        
        # Wall is between the two cells
        wall_row = (gr1 + gr2) // 2
        wall_col = (gc1 + gc2) // 2
        
        # Mark cells and wall as path (False)
        self.grid[gr1, gc1] = False
        self.grid[gr2, gc2] = False
        self.grid[wall_row, wall_col] = False
    
    def generate(self, visualize: bool = False, delay: float = 0.01) -> np.ndarray:
        """
        Generate the maze using recursive backtracker algorithm.
        
        Args:
            visualize: Whether to show real-time visualization with Pygame
            delay: Delay between steps in seconds (for visualization)
            
        Returns:
            2D boolean array where True=wall, False=path
        """
        # Start from top-left cell (0, 0)
        start_cell = (0, 0)
        self.visited[start_cell] = True
        self.stack.append(start_cell)
        self.current_cell = start_cell
        
        # Mark start cell as path
        gr, gc = self._cell_to_grid(0, 0)
        self.grid[gr, gc] = False
        
        # Initialize Pygame if visualizing
        if visualize:
            pygame.init()
            cell_size = 30  # Size of each cell in pixels
            wall_thickness = cell_size // 5  # Walls are 1/5 of cell width
            
            # Calculate screen size based on grid
            screen_width = self.width * cell_size + (self.width + 1) * wall_thickness
            screen_height = self.height * cell_size + (self.height + 1) * wall_thickness
            
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Maze Generator - Recursive Backtracker")
            clock = pygame.time.Clock()
            
            # Colors
            BLACK = (0, 0, 0)  # Unvisited cells
            DARK_GREEN = (0, 100, 0)  # Explored path
            BRIGHT_GREEN = (0, 255, 0)  # Walls
            WHITE = (255, 255, 255)  # Current position
        
        # Recursive backtracker algorithm
        while self.stack:
            if visualize:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return self.grid
            
            current = self.stack[-1]
            self.current_cell = current
            neighbors = self._get_neighbors(current[0], current[1])
            
            if neighbors:
                # Choose random neighbor
                next_cell = random.choice(neighbors)
                
                # Remove wall between current and next
                self._remove_wall(current, next_cell)
                
                # Mark as visited and push to stack
                self.visited[next_cell] = True
                self.stack.append(next_cell)
                
            else:
                # Backtrack
                self.stack.pop()
            
            # Visualize current state
            if visualize:
                screen.fill(BLACK)
                
                # Draw the maze
                for row in range(2 * self.height + 1):
                    for col in range(2 * self.width + 1):
                        # Calculate pixel position
                        if row % 2 == 1 and col % 2 == 1:
                            # This is a cell position
                            cell_row = row // 2
                            cell_col = col // 2
                            
                            x = cell_col * (cell_size + wall_thickness) + wall_thickness
                            y = cell_row * (cell_size + wall_thickness) + wall_thickness
                            
                            if self.visited[cell_row, cell_col]:
                                # Explored cell - dark green
                                color = DARK_GREEN
                            else:
                                # Unvisited cell - black
                                color = BLACK
                            
                            pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))
                            
                            # Draw current position as white square filling whole cell
                            if self.current_cell == (cell_row, cell_col):
                                pygame.draw.rect(screen, WHITE, (x, y, cell_size, cell_size))
                        
                        elif row % 2 == 0 or col % 2 == 0:
                            # This is a wall position
                            if self.grid[row, col]:
                                # Wall exists - bright green
                                if row % 2 == 0 and col % 2 == 0:
                                    # Corner
                                    if col % 2 == 0:
                                        x = (col // 2) * (cell_size + wall_thickness)
                                    if row % 2 == 0:
                                        y = (row // 2) * (cell_size + wall_thickness)
                                    pygame.draw.rect(screen, BRIGHT_GREEN, 
                                                   (x, y, wall_thickness, wall_thickness))
                                elif row % 2 == 0:
                                    # Horizontal wall
                                    cell_col = col // 2
                                    x = cell_col * (cell_size + wall_thickness) + wall_thickness
                                    y = (row // 2) * (cell_size + wall_thickness)
                                    pygame.draw.rect(screen, BRIGHT_GREEN, 
                                                   (x, y, cell_size, wall_thickness))
                                else:
                                    # Vertical wall
                                    cell_row = row // 2
                                    x = (col // 2) * (cell_size + wall_thickness)
                                    y = cell_row * (cell_size + wall_thickness) + wall_thickness
                                    pygame.draw.rect(screen, BRIGHT_GREEN, 
                                                   (x, y, wall_thickness, cell_size))
                            else:
                                # Wall removed - show dark green path
                                if row % 2 == 0 and col % 2 == 0:
                                    # Corner where walls removed
                                    if col % 2 == 0:
                                        x = (col // 2) * (cell_size + wall_thickness)
                                    if row % 2 == 0:
                                        y = (row // 2) * (cell_size + wall_thickness)
                                    pygame.draw.rect(screen, DARK_GREEN, 
                                                   (x, y, wall_thickness, wall_thickness))
                                elif row % 2 == 0:
                                    # Horizontal wall removed
                                    cell_col = col // 2
                                    x = cell_col * (cell_size + wall_thickness) + wall_thickness
                                    y = (row // 2) * (cell_size + wall_thickness)
                                    pygame.draw.rect(screen, DARK_GREEN, 
                                                   (x, y, cell_size, wall_thickness))
                                else:
                                    # Vertical wall removed
                                    cell_row = row // 2
                                    x = (col // 2) * (cell_size + wall_thickness)
                                    y = cell_row * (cell_size + wall_thickness) + wall_thickness
                                    pygame.draw.rect(screen, DARK_GREEN, 
                                                   (x, y, wall_thickness, cell_size))
                
                pygame.display.flip()
                clock.tick(1 / delay)  # Control speed
        
        if visualize:
            # Final display - redraw one more time to show complete maze
            screen.fill(BLACK)
            
            for row in range(2 * self.height + 1):
                for col in range(2 * self.width + 1):
                    if row % 2 == 1 and col % 2 == 1:
                        # Cell position
                        cell_row = row // 2
                        cell_col = col // 2
                        
                        x = cell_col * (cell_size + wall_thickness) + wall_thickness
                        y = cell_row * (cell_size + wall_thickness) + wall_thickness
                        
                        # All cells are explored now - dark green
                        pygame.draw.rect(screen, DARK_GREEN, (x, y, cell_size, cell_size))
                    
                    elif row % 2 == 0 or col % 2 == 0:
                        # Wall position
                        if self.grid[row, col]:
                            # Wall exists - bright green
                            if row % 2 == 0 and col % 2 == 0:
                                x = (col // 2) * (cell_size + wall_thickness)
                                y = (row // 2) * (cell_size + wall_thickness)
                                pygame.draw.rect(screen, BRIGHT_GREEN, 
                                               (x, y, wall_thickness, wall_thickness))
                            elif row % 2 == 0:
                                cell_col = col // 2
                                x = cell_col * (cell_size + wall_thickness) + wall_thickness
                                y = (row // 2) * (cell_size + wall_thickness)
                                pygame.draw.rect(screen, BRIGHT_GREEN, 
                                               (x, y, cell_size, wall_thickness))
                            else:
                                cell_row = row // 2
                                x = (col // 2) * (cell_size + wall_thickness)
                                y = cell_row * (cell_size + wall_thickness) + wall_thickness
                                pygame.draw.rect(screen, BRIGHT_GREEN, 
                                               (x, y, wall_thickness, cell_size))
                        else:
                            # Wall removed - dark green
                            if row % 2 == 0 and col % 2 == 0:
                                x = (col // 2) * (cell_size + wall_thickness)
                                y = (row // 2) * (cell_size + wall_thickness)
                                pygame.draw.rect(screen, DARK_GREEN, 
                                               (x, y, wall_thickness, wall_thickness))
                            elif row % 2 == 0:
                                cell_col = col // 2
                                x = cell_col * (cell_size + wall_thickness) + wall_thickness
                                y = (row // 2) * (cell_size + wall_thickness)
                                pygame.draw.rect(screen, DARK_GREEN, 
                                               (x, y, cell_size, wall_thickness))
                            else:
                                cell_row = row // 2
                                x = (col // 2) * (cell_size + wall_thickness)
                                y = cell_row * (cell_size + wall_thickness) + wall_thickness
                                pygame.draw.rect(screen, DARK_GREEN, 
                                               (x, y, wall_thickness, cell_size))
            
            pygame.display.flip()
            
            # Keep window open - wait for user to close it manually
            print("\nMaze generation complete! Close the Pygame window to continue...")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
            
            pygame.quit()
        
        return self.grid
    
    def get_start_position(self) -> Tuple[int, int]:
        """Get the start position (top-left in grid coordinates)."""
        return self._cell_to_grid(0, 0)
    
    def get_random_goal_position(self, seed: Optional[int] = None) -> Tuple[int, int]:
        """
        Get a random goal position that's not the start.
        
        Args:
            seed: Optional seed for goal randomization
            
        Returns:
            Grid coordinates of goal position
        """
        if seed is not None:
            random.seed(seed)
        
        # Choose random cell that's not the start
        while True:
            goal_row = random.randint(0, self.height - 1)
            goal_col = random.randint(0, self.width - 1)
            
            if (goal_row, goal_col) != (0, 0):
                return self._cell_to_grid(goal_row, goal_col)
    
    def save_maze(self, filename: str):
        """Save the maze to a numpy file."""
        np.save(filename, self.grid)
        print(f"Maze saved to {filename}")
    
    def save_maze_to_txt(self, filename: str, start: Tuple[int, int] = None, goal: Tuple[int, int] = None):
        """
        Save the maze to a text file.
        Uses '1' for walls, '0' for paths, '3' for start, and '4' for goal.
        Numbers are space-separated.
        
        Args:
            filename: Output filename
            start: Start position in grid coordinates (if None, will be top-left)
            goal: Goal position in grid coordinates (if None, will be random)
        """
        if start is None:
            start = self.get_start_position()
        if goal is None:
            goal = self.get_random_goal_position()
        
        with open(filename, 'w') as f:
            for row_idx, row in enumerate(self.grid):
                row_values = []
                for col_idx, cell in enumerate(row):
                    # Check if this is the start position
                    if (row_idx, col_idx) == start:
                        row_values.append('3')
                    # Check if this is the goal position
                    elif (row_idx, col_idx) == goal:
                        row_values.append('4')
                    else:
                        # 1 for wall, 0 for path
                        row_values.append('1' if cell else '0')
                
                # Join with spaces and write
                f.write(' '.join(row_values) + '\n')
        
        print(f"Maze saved to text file: {filename}")
        print(f"  Format: 1=wall, 0=path, 3=start, 4=goal")
        print(f"  Start position: {start}")
        print(f"  Goal position: {goal}")
    
    def visualize_and_save_png(self, filename: str = "maze_output.png", start: Tuple[int, int] = None, goal: Tuple[int, int] = None):
        """
        Generate and save a PNG visualization matching the Pygame style.
        Black background, green walls, white start (full cell), red goal (full cell).
        
        Args:
            filename: Output filename for the PNG
            start: Start position in grid coordinates (default: top-left)
            goal: Goal position in grid coordinates (default: random)
        """
        if start is None:
            start = self.get_start_position()
        if goal is None:
            goal = self.get_random_goal_position()
        
        # Calculate dimensions
        cell_size = 30
        wall_thickness = cell_size // 5
        
        # Create figure with black background
        height_px = self.height * cell_size + (self.height + 1) * wall_thickness
        width_px = self.width * cell_size + (self.width + 1) * wall_thickness
        
        dpi = 100
        fig, ax = plt.subplots(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
        ax.set_xlim(0, width_px)
        ax.set_ylim(0, height_px)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Black background
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Colors matching Pygame
        BLACK = (0, 0, 0)
        DARK_GREEN = (0, 100/255, 0)
        BRIGHT_GREEN = (0, 1, 0)
        WHITE = (1, 1, 1)
        RED = (1, 0, 0)
        
        # Draw the maze
        for row in range(2 * self.height + 1):
            for col in range(2 * self.width + 1):
                if row % 2 == 1 and col % 2 == 1:
                    # Cell position
                    cell_row = row // 2
                    cell_col = col // 2
                    
                    x = col // 2 * (cell_size + wall_thickness) + wall_thickness
                    y = height_px - (cell_row * (cell_size + wall_thickness) + wall_thickness) - cell_size
                    
                    # All cells are dark green (explored)
                    rect = plt.Rectangle((x, y), cell_size, cell_size, 
                                        facecolor=DARK_GREEN, edgecolor='none')
                    ax.add_patch(rect)
                    
                    # Mark start as white square
                    if (row, col) == (start[0], start[1]):
                        rect = plt.Rectangle((x, y), cell_size, cell_size, 
                                            facecolor=WHITE, edgecolor='none')
                        ax.add_patch(rect)
                    
                    # Mark goal as red square
                    if (row, col) == (goal[0], goal[1]):
                        rect = plt.Rectangle((x, y), cell_size, cell_size, 
                                            facecolor=RED, edgecolor='none')
                        ax.add_patch(rect)
                
                elif row % 2 == 0 or col % 2 == 0:
                    # Wall position
                    if self.grid[row, col]:
                        # Wall exists - bright green
                        if row % 2 == 0 and col % 2 == 0:
                            # Corner
                            x = (col // 2) * (cell_size + wall_thickness)
                            y = height_px - ((row // 2) * (cell_size + wall_thickness)) - wall_thickness
                            rect = plt.Rectangle((x, y), wall_thickness, wall_thickness,
                                               facecolor=BRIGHT_GREEN, edgecolor='none')
                            ax.add_patch(rect)
                        elif row % 2 == 0:
                            # Horizontal wall
                            cell_col = col // 2
                            x = cell_col * (cell_size + wall_thickness) + wall_thickness
                            y = height_px - ((row // 2) * (cell_size + wall_thickness)) - wall_thickness
                            rect = plt.Rectangle((x, y), cell_size, wall_thickness,
                                               facecolor=BRIGHT_GREEN, edgecolor='none')
                            ax.add_patch(rect)
                        else:
                            # Vertical wall
                            cell_row = row // 2
                            x = (col // 2) * (cell_size + wall_thickness)
                            y = height_px - (cell_row * (cell_size + wall_thickness) + wall_thickness) - cell_size
                            rect = plt.Rectangle((x, y), wall_thickness, cell_size,
                                               facecolor=BRIGHT_GREEN, edgecolor='none')
                            ax.add_patch(rect)
                    else:
                        # Wall removed - dark green
                        if row % 2 == 0 and col % 2 == 0:
                            x = (col // 2) * (cell_size + wall_thickness)
                            y = height_px - ((row // 2) * (cell_size + wall_thickness)) - wall_thickness
                            rect = plt.Rectangle((x, y), wall_thickness, wall_thickness,
                                               facecolor=DARK_GREEN, edgecolor='none')
                            ax.add_patch(rect)
                        elif row % 2 == 0:
                            cell_col = col // 2
                            x = cell_col * (cell_size + wall_thickness) + wall_thickness
                            y = height_px - ((row // 2) * (cell_size + wall_thickness)) - wall_thickness
                            rect = plt.Rectangle((x, y), cell_size, wall_thickness,
                                               facecolor=DARK_GREEN, edgecolor='none')
                            ax.add_patch(rect)
                        else:
                            cell_row = row // 2
                            x = (col // 2) * (cell_size + wall_thickness)
                            y = height_px - (cell_row * (cell_size + wall_thickness) + wall_thickness) - cell_size
                            rect = plt.Rectangle((x, y), wall_thickness, cell_size,
                                               facecolor=DARK_GREEN, edgecolor='none')
                            ax.add_patch(rect)
        
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()
        print(f"Maze visualization saved to: {filename}")
    
    def load_maze(self, filename: str):
        """Load a maze from a numpy file."""
        self.grid = np.load(filename)
        print(f"Maze loaded from {filename}")
    
    def validate_maze(self) -> bool:
        """
        Validate that the maze is perfect (tree property).
        Number of removed walls should equal (W*H - 1).
        """
        # Count path cells (excluding walls)
        path_cells = 0
        for row in range(1, 2 * self.height + 1, 2):
            for col in range(1, 2 * self.width + 1, 2):
                if not self.grid[row, col]:
                    path_cells += 1
        
        # Count removed walls (horizontal and vertical)
        removed_walls = 0
        
        # Horizontal walls
        for row in range(1, 2 * self.height + 1, 2):
            for col in range(2, 2 * self.width, 2):
                if not self.grid[row, col]:
                    removed_walls += 1
        
        # Vertical walls
        for row in range(2, 2 * self.height, 2):
            for col in range(1, 2 * self.width + 1, 2):
                if not self.grid[row, col]:
                    removed_walls += 1
        
        expected_walls = self.width * self.height - 1
        is_valid = removed_walls == expected_walls
        
        print(f"Validation: {path_cells} cells, {removed_walls} removed walls")
        print(f"Expected: {expected_walls} removed walls")
        print(f"Maze is {'VALID' if is_valid else 'INVALID'}")
        
        return is_valid


def main():
    """
    Main function to generate maze with custom configuration.
    Edit the width and height variables below to change maze size.
    """
    print("=== Maze Generator ===\n")
    
    # ============================================
    # CONFIGURATION - Edit these values!
    # ============================================
    width = 5   # Number of cells horizontally
    height = 5  # Number of cells vertically
    # ============================================
    
    # Use current time as seed for random generation
    seed = int(time.time())
    
    print(f"Generating {width}x{height} maze")
    print(f"Using random seed: {seed}")
    print(f"Grid dimensions: {2*height+1} x {2*width+1}")
    print("Starting from top-left corner...\n")
    
    # Create generator with random seed
    generator = MazeGenerator(width, height, seed)
    
    # Generate with real-time visualization
    print("Watch the real-time visualization!")
    maze = generator.generate(visualize=True, delay=0.02)
    
    # Validate the maze
    print("\n" + "="*50)
    print("Validating maze structure...")
    generator.validate_maze()
    
    # Get start and goal positions
    start = generator.get_start_position()
    goal = generator.get_random_goal_position()
    
    print(f"\nStart position (grid coords): {start}")
    print(f"Goal position (grid coords): {goal}")
    print(f"Maze shape: {maze.shape}")
    
    # Save outputs
    print("\n" + "="*50)
    print("Saving maze outputs...")
    
    # Save as numpy array
    generator.save_maze("maze_output.npy")
    
    # Save as text file (with start marked as 3 and goal marked as 4)
    generator.save_maze_to_txt("maze.txt", start, goal)
    
    # Save as PNG visualization
    generator.visualize_and_save_png("maze_output.png", start, goal)
    
    print("\n" + "="*50)
    print("✓ Maze generation complete!")
    print("="*50)
    print("\nGenerated files:")
    print("  - maze_output.npy  (NumPy array format)")
    print("  - maze.txt         (Text format: 1=wall, 0=path, 3=start, 4=goal)")
    print("  - maze_output.png  (PNG visualization)")
    print("\nStart = White square (marked as 3 in txt)")
    print("Goal = Red square (marked as 4 in txt)")
    

if __name__ == "__main__":
    main()