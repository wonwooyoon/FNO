import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def generate_scp_visualization_data(N, b, size_x, size_y, level_max, density_map_ratio):
    """
    Generate SCP field data while tracking all levels for visualization.

    Args:
        N: Number of squares per parent square
        b: Subdivision factor
        size_x: Width of rectangular domain
        size_y: Height of rectangular domain
        level_max: Maximum recursion level
        density_map_ratio: Ratio for density map resolution

    Returns:
        level_coords: Dictionary mapping level -> list of (x, y, side_length) tuples
        density_map: Final density map array
    """
    # Calculate density map dimensions based on ratio
    density_map_size_x = int(size_x * density_map_ratio)
    density_map_size_y = int(size_y * density_map_ratio)
    density_map = np.ones((density_map_size_x, density_map_size_y))

    # Store coordinates for each level
    level_coords = {level: [] for level in range(1, level_max + 1)}

    def draw_squares(N, b, size_x, size_y, level=1, parent_coords=[(0, 0)]):
        """Recursively draw squares and track coordinates at each level."""

        if level > level_max:
            return

        # Calculate side_length as average of dimensions divided by b^level
        side_length = (size_x + size_y) / 2 / (b ** level)
        new_coords = []

        for px, py in parent_coords:
            for _ in range(N):
                if level == 1:  # First level: distribute across entire domain
                    x = random.uniform(0, size_x)
                    y = random.uniform(0, size_y)
                else:  # Subsequent levels: fractal subdivision
                    x = px + random.uniform(0, side_length * b)
                    y = py + random.uniform(0, side_length * b)

                # Handle wrapping for rectangular domain
                if x >= size_x:
                    x = x - size_x
                if y >= size_y:
                    y = y - size_y

                new_coords.append((x, y))

                # Store coordinates for visualization
                level_coords[level].append((x, y, side_length))

                if level == level_max:
                    update_density_map(density_map, x, y, side_length, size_x, size_y,
                                     density_map_size_x, density_map_size_y)

        draw_squares(N, b, size_x, size_y, level + 1, new_coords)

    def update_density_map(density_map, x, y, side_length, size_x, size_y,
                          density_map_size_x, density_map_size_y):
        """Update density map by incrementing cells covered by square."""

        for i in range(int(side_length)):
            for j in range(int(side_length)):

                # Calculate grid indices for rectangular domain
                xi = int((x + i) // (size_x / density_map_size_x))
                yj = int((y + j) // (size_y / density_map_size_y))

                # Handle wrapping for rectangular density map
                if xi > density_map_size_x - 1:
                    xi = xi - density_map_size_x
                if yj > density_map_size_y - 1:
                    yj = yj - density_map_size_y

                density_map[xi, yj] += 1

    # Execute SCP algorithm
    draw_squares(N, b, size_x, size_y)

    return level_coords, density_map


def visualize_level_squares(level_coords, level, size_x, size_y, output_dir):
    """
    Visualize squares at a specific level with previous levels shown in different colors.

    Args:
        level_coords: Dictionary of coordinates per level
        level: Level to visualize
        size_x: Domain width
        size_y: Domain height
        output_dir: Output directory path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color map for different levels
    colors = ["#B8474B", '#4bab72', "#5164D1BC", "#e1ff00"]

    # Draw all levels up to current level
    for lv in range(1, level + 1):
        coords = level_coords[lv]
        color_idx = (lv - 1) % len(colors)

        for x, y, side_length in coords:
            # Handle wrapping: if square extends beyond boundary, draw wrapped portions
            # Check if square wraps in X direction
            if x + side_length > size_x:
                # Split into two parts: right part and wrapped left part
                right_width = size_x - x
                left_width = side_length - right_width

                # Check if also wraps in Y direction
                if y + side_length > size_y:
                    # Wraps in both X and Y: draw 4 rectangles
                    bottom_height = size_y - y
                    top_height = side_length - bottom_height

                    # Bottom-right
                    rect1 = patches.Rectangle((x, y), right_width, bottom_height,
                                             linewidth=1.0, edgecolor='black',
                                             facecolor=colors[color_idx], alpha=0.5)
                    ax.add_patch(rect1)
                    # Bottom-left (wrapped X)
                    rect2 = patches.Rectangle((0, y), left_width, bottom_height,
                                             linewidth=1.0, edgecolor='black',
                                             facecolor=colors[color_idx], alpha=0.5)
                    ax.add_patch(rect2)
                    # Top-right (wrapped Y)
                    rect3 = patches.Rectangle((x, 0), right_width, top_height,
                                             linewidth=1.0, edgecolor='black',
                                             facecolor=colors[color_idx], alpha=0.5)
                    ax.add_patch(rect3)
                    # Top-left (wrapped both)
                    rect4 = patches.Rectangle((0, 0), left_width, top_height,
                                             linewidth=1.0, edgecolor='black',
                                             facecolor=colors[color_idx], alpha=0.5)
                    ax.add_patch(rect4)
                else:
                    # Wraps only in X: draw 2 rectangles
                    # Right part
                    rect1 = patches.Rectangle((x, y), right_width, side_length,
                                             linewidth=1.0, edgecolor='black',
                                             facecolor=colors[color_idx], alpha=0.5)
                    ax.add_patch(rect1)
                    # Left part (wrapped)
                    rect2 = patches.Rectangle((0, y), left_width, side_length,
                                             linewidth=1.0, edgecolor='black',
                                             facecolor=colors[color_idx], alpha=0.5)
                    ax.add_patch(rect2)
            elif y + side_length > size_y:
                # Wraps only in Y: draw 2 rectangles
                bottom_height = size_y - y
                top_height = side_length - bottom_height

                # Bottom part
                rect1 = patches.Rectangle((x, y), side_length, bottom_height,
                                         linewidth=1.0, edgecolor='black',
                                         facecolor=colors[color_idx], alpha=0.5)
                ax.add_patch(rect1)
                # Top part (wrapped)
                rect2 = patches.Rectangle((x, 0), side_length, top_height,
                                         linewidth=1.0, edgecolor='black',
                                         facecolor=colors[color_idx], alpha=0.5)
                ax.add_patch(rect2)
            else:
                # No wrapping: draw single rectangle
                rect = patches.Rectangle((x, y), side_length, side_length,
                                         linewidth=1.0, edgecolor='black',
                                         facecolor=colors[color_idx], alpha=0.5)
                ax.add_patch(rect)

    ax.set_xlim(0, size_x)
    ax.set_ylim(0, size_y)
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axes, labels, ticks

    output_path = os.path.join(output_dir, f'level_{level}_squares.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {output_path}")


def visualize_density_map(density_map, density_map_ratio, output_dir):
    """
    Visualize density map as heatmap with correct dimensions.

    Args:
        density_map: Density map array
        density_map_ratio: Ratio for density map resolution
        output_dir: Output directory path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get actual density map dimensions
    density_map_size_x, density_map_size_y = density_map.shape

    ax.imshow(density_map.T, origin='lower', cmap='hot',
              extent=[0, density_map_size_x, 0, density_map_size_y], aspect='auto')
    ax.axis('off')  # Remove axes, labels, ticks

    output_path = os.path.join(output_dir, 'density_map.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {output_path}")


def visualize_perm_map(density_map, output_dir):
    """
    Generate and visualize permeability map in log10 scale.

    Args:
        density_map: Density map array
        output_dir: Output directory path
    """
    # Generate permeability map (same logic as initial_perm.py)
    average_aperature = np.random.uniform(0.00001, 0.0001)

    domain_size = density_map.shape[0] * density_map.shape[1]
    total_density = np.sum(density_map)
    aperature_ratio = average_aperature * domain_size / total_density
    aperature_map = np.minimum(0.005, aperature_ratio * density_map)

    perm_map = aperature_map ** 3 / 12.0 / 100.0 / 0.005

    # Apply log10 transform for visualization
    perm_map_log = np.log10(perm_map + 1e-20)  # Add small value to avoid log(0)

    # Get density map dimensions
    density_map_size_x, density_map_size_y = density_map.shape

    # Visualize permeability map in log10 scale
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(perm_map_log.T, origin='lower', cmap='viridis',
              extent=[0, density_map_size_x, 0, density_map_size_y], aspect='auto')
    ax.axis('off')  # Remove axes, labels, ticks

    output_path = os.path.join(output_dir, 'permeability_map.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """
    Main workflow:
    1. Set SCP parameters
    2. Generate SCP data with level tracking
    3. Visualize each level's squares (cumulative with different colors)
    4. Visualize density map
    5. Visualize permeability map (log10 scale)
    """
    # SCP parameters
    CONFIG = {
        'N': 9,
        'b': 2.64,
        'size_x': 256,
        'size_y': 128,
        'level_max': 3,
        'density_map_ratio': 0.25,
        'output_dir': '/home/geofluids/research/FNO/src/initial_perm/output_visualization'
    }

    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Output directory: {CONFIG['output_dir']}")

    # Set random seed for reproducibility
    random.seed(72)
    np.random.seed(72)

    # Generate SCP data
    print("\nGenerating SCP field data...")
    level_coords, density_map = generate_scp_visualization_data(
        CONFIG['N'], CONFIG['b'], CONFIG['size_x'], CONFIG['size_y'],
        CONFIG['level_max'], CONFIG['density_map_ratio']
    )

    # Visualize each level's squares (cumulative)
    print("\nVisualizing level-by-level squares...")
    for level in range(1, CONFIG['level_max'] + 1):
        visualize_level_squares(level_coords, level, CONFIG['size_x'],
                               CONFIG['size_y'], CONFIG['output_dir'])

    # Visualize density map
    print("\nVisualizing density map...")
    visualize_density_map(density_map, CONFIG['density_map_ratio'],
                         CONFIG['output_dir'])

    # Visualize permeability map
    print("\nVisualizing permeability map...")
    visualize_perm_map(density_map, CONFIG['output_dir'])

    print("\n✓ All visualizations completed!")


if __name__ == '__main__':
    main()
