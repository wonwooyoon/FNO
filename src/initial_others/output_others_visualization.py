import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_normalized_lhs(csv_path, output_dir):
    """
    Visualize normalized Latin Hypercube Sampling in 3D space.

    Args:
        csv_path: Path to the others.csv file
        output_dir: Directory to save the visualization
    """
    # Read CSV data
    df = pd.read_csv(csv_path)

    # Extract columns
    pressure = df['pressure'].values
    degra_mont = df['degra_mont'].values
    ratio = df['ratio'].values

    print(f"Loaded {len(df)} samples")
    print(f"Pressure range: [{pressure.min():.2f}, {pressure.max():.2f}]")
    print(f"Degra_mont range: [{degra_mont.min():.4f}, {degra_mont.max():.4f}]")
    print(f"Ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")

    # Create normalized 3D plot with adjusted size
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize to [0, 1]
    pressure_norm = (pressure - pressure.min()) / (pressure.max() - pressure.min())
    degra_mont_norm = (degra_mont - degra_mont.min()) / (degra_mont.max() - degra_mont.min())
    ratio_norm = (ratio - ratio.min()) / (ratio.max() - ratio.min())

    # Scatter plot
    ax.scatter(pressure_norm, degra_mont_norm, ratio_norm,
               c=ratio_norm, cmap='viridis',
               marker='o', s=20, alpha=0.6,
               edgecolors='k', linewidth=0.3)

    # Set labels with mathematical notation
    # N: script font (\mathcal{N})
    # nabla: gradient symbol (\nabla)
    # rho: Greek letter (\rho)
    # subscripts: _{text}
    ax.set_xlabel(r'$\mathcal{N}(\nabla p)$', fontsize=18, labelpad=12)
    ax.set_ylabel(r'$\mathcal{N}(\rho_{\mathrm{bnt}})$', fontsize=18, labelpad=12)
    ax.set_zlabel(r'$\mathcal{N}(f_{\mathrm{mixing}})$', fontsize=18, labelpad=12)

    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)

    # Adjust subplot to make room for labels
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Save plot with tight layout but with padding
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lhs_sampling_normalized.png')
    plt.savefig(output_path, dpi=600, pad_inches=0.3)
    print(f"Saved normalized visualization to: {output_path}")
    plt.close()


def main():
    """
    Main workflow:
    1. Load others.csv data
    2. Create normalized 3D scatter plot
    3. Save visualization
    """
    # Configuration
    csv_path = '/home/geofluids/research/FNO/src/initial_others/output/others.csv'
    output_dir = '/home/geofluids/research/FNO/src/initial_others/output_visualization'

    print("Starting Latin Hypercube Sampling visualization...")
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")

    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Generate normalized visualization
    visualize_normalized_lhs(csv_path, output_dir)

    print("\n✓ Visualization completed!")


if __name__ == '__main__':
    main()
