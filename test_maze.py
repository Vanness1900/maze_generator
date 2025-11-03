"""
Simple test script to verify the maze generator works correctly.
Run this after installing dependencies to check everything is set up properly.
"""

from maze_generator import MazeGenerator
import numpy as np


def test_basic_generation():
    """Test basic maze generation."""
    print("Test 1: Basic maze generation...")
    gen = MazeGenerator(5, 5, seed=1)
    maze = gen.generate(visualize=False)
    assert maze.shape == (11, 11), "Incorrect maze shape"
    print("✓ Basic generation works")


def test_validation():
    """Test maze validation."""
    print("\nTest 2: Maze validation...")
    gen = MazeGenerator(8, 8, seed=2)
    maze = gen.generate(visualize=False)
    is_valid = gen.validate_maze()
    assert is_valid, "Maze validation failed"
    print("✓ Validation works")


def test_reproducibility():
    """Test that same seed produces same maze."""
    print("\nTest 3: Reproducibility...")
    gen1 = MazeGenerator(6, 6, seed=42)
    maze1 = gen1.generate(visualize=False)
    
    gen2 = MazeGenerator(6, 6, seed=42)
    maze2 = gen2.generate(visualize=False)
    
    assert np.array_equal(maze1, maze2), "Mazes differ with same seed"
    print("✓ Reproducibility works")


def test_start_goal():
    """Test start and goal position generation."""
    print("\nTest 4: Start and goal positions...")
    gen = MazeGenerator(10, 10, seed=3)
    maze = gen.generate(visualize=False)
    
    start = gen.get_start_position()
    goal = gen.get_random_goal_position(seed=4)
    
    assert start == (1, 1), "Start position incorrect"
    assert start != goal, "Start and goal are the same"
    print(f"✓ Start: {start}, Goal: {goal}")


def test_save_load():
    """Test saving and loading mazes."""
    print("\nTest 5: Save and load...")
    gen1 = MazeGenerator(7, 7, seed=5)
    maze1 = gen1.generate(visualize=False)
    
    gen1.save_maze("test_temp.npy")
    
    gen2 = MazeGenerator(7, 7)
    gen2.load_maze("test_temp.npy")
    
    assert np.array_equal(maze1, gen2.grid), "Loaded maze differs from saved"
    print("✓ Save and load works")
    
    # Clean up
    import os
    if os.path.exists("test_temp.npy"):
        os.remove("test_temp.npy")


def test_different_sizes():
    """Test various maze sizes."""
    print("\nTest 6: Different maze sizes...")
    sizes = [(3, 3), (10, 5), (15, 15)]
    
    for width, height in sizes:
        gen = MazeGenerator(width, height, seed=6)
        maze = gen.generate(visualize=False)
        expected_shape = (2 * height + 1, 2 * width + 1)
        assert maze.shape == expected_shape, f"Wrong shape for {width}x{height}"
        assert gen.validate_maze(), f"Invalid maze for {width}x{height}"
    
    print(f"✓ Tested {len(sizes)} different sizes")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Maze Generator Tests")
    print("=" * 50)
    
    try:
        test_basic_generation()
        test_validation()
        test_reproducibility()
        test_start_goal()
        test_save_load()
        test_different_sizes()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYour maze generator is working correctly!")
        print("Run 'python maze_generator.py' to see the visualization demo.")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)