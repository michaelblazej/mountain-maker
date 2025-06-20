#!/usr/bin/env python3
"""
Visualize Mountain Maker DLA grid CSV files using matplotlib.

Usage:
    python visualize.py [grid_file.txt] [output_image.png]
    
If no arguments are provided, it will look for dla_base_grid.txt
"""

import sys
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def read_grid_csv(filepath):
    """Read a grid from the CSV file format exported by mountain-maker."""
    data = []
    width = 0
    height = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                # Try to extract grid dimensions from header comment
                if "Grid dimensions:" in line:
                    try:
                        parts = line.split(":")
                        dims = parts[1].strip().split("x")
                        width = int(dims[0].strip())
                        height = int(dims[1].strip())
                    except Exception as e:
                        print(f"Warning: Could not parse grid dimensions: {e}")
                continue
                
            # Parse CSV values as integers
            try:
                row = [float(val) for val in line.strip().split(',')]
                data.append(row)
            except Exception as e:
                print(f"Warning: Error parsing line: {e}")
                continue
    
    # Convert to numpy array
    grid = np.array(data)
    return grid

def visualize_grid(grid, output_file=None, title=None, cmap='viridis'):
    """Visualize a grid using matplotlib."""
    # Create a figure with a larger size for good resolution
    plt.figure(figsize=(12, 12))
    
    # Determine a good colormap based on data
    if np.all(np.logical_or(grid == 0, grid == 1)):
        # Binary data, use a custom colormap
        colors = [(1, 1, 1, 0), (0, 0.3, 0.7, 1)]  # Transparent white to blue
        cmap = LinearSegmentedColormap.from_list("binary_blue", colors)
    
    # Plot the grid with the chosen colormap
    plt.imshow(grid, cmap=cmap, interpolation='none')
    
    # Remove axes for cleaner look
    plt.axis('off')
    
    # Set title if provided
    if title:
        plt.title(title)
    
    # Add colorbar for reference
    plt.colorbar(shrink=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    # Show the plot
    plt.show()

def main():
    # Default input and output files
    input_file = "dla_base_grid.txt"
    output_file = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return 1
    
    # Generate output filename if not specified
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.png"
    
    # Read the grid
    try:
        grid = read_grid_csv(input_file)
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return 1
    
    # Generate title from filename
    title = os.path.basename(input_file)
    
    # Visualize
    visualize_grid(grid, output_file, title)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
