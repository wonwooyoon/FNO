#!/usr/bin/env python3
"""
Compare LR and HR permeability maps in Fourier domain using forward normalization

Compares perm_map_0.h5 files from output_lr and output_hr directories.
Assumes both represent the same 16m × 8m physical domain at different resolutions.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_perm_map(filepath, apply_log=False, epsilon=1e-20):
    """
    Load permeability map from HDF5 file

    Args:
        filepath: Path to HDF5 file
        apply_log: If True, apply log transformation
        epsilon: Small value to add before log to avoid log(0)
    """
    with h5py.File(filepath, 'r') as f:
        perm_map = f['permsX']['Data'][:]
        discretization = f['permsX'].attrs['Discretization']
        origin = f['permsX'].attrs['Origin']

    if apply_log:
        perm_map = np.log(perm_map + epsilon)

    return perm_map, discretization, origin


def compute_fft_modes(data, domain_size):
    """
    Compute 2D FFT with forward normalization

    Args:
        data: 2D array
        domain_size: (Lx, Ly) physical domain size in meters

    Returns:
        fft_magnitude: FFT magnitude (shifted to center)
        mode_x: X mode indices
        mode_y: Y mode indices
    """
    nx, ny = data.shape
    Lx, Ly = domain_size

    # FFT with forward normalization and grid spacing correction
    fft_result = np.fft.fft2(data, norm='forward')
    fft_shifted = np.fft.fftshift(fft_result)
    fft_magnitude = np.abs(fft_shifted)

    # FFT with forward normalization (same as neuralop FNO)
    # No additional grid spacing normalization - matches FNO implementation
    fft_result = np.fft.fft2(data, norm='forward')
    fft_shifted = np.fft.fftshift(fft_result)
    fft_magnitude = np.abs(fft_shifted)

    # Mode indices
    mode_x = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0/nx))
    mode_y = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0/ny))

    return fft_magnitude, mode_x, mode_y


def align_spectra(lr_mag, lr_mode_x, lr_mode_y, hr_mag, hr_mode_x, hr_mode_y):
    """Align HR spectrum to LR modes for direct comparison"""
    # Find HR indices corresponding to LR modes
    hr_idx_x = [np.argmin(np.abs(hr_mode_x - mode)) for mode in lr_mode_x]
    hr_idx_y = [np.argmin(np.abs(hr_mode_y - mode)) for mode in lr_mode_y]

    aligned_hr_mag = hr_mag[np.ix_(hr_idx_x, hr_idx_y)]

    return lr_mag, aligned_hr_mag


def extract_1d_spectrum(fft_mag, mode_x, mode_y):
    """Extract 1D spectra along X and Y axes (at mode=0 for other dimension)"""
    zero_idx_x = np.argmin(np.abs(mode_x))
    zero_idx_y = np.argmin(np.abs(mode_y))

    spectrum_along_x = fft_mag[:, zero_idx_y]  # Varying kx, ky=0
    spectrum_along_y = fft_mag[zero_idx_x, :]  # Varying ky, kx=0

    return spectrum_along_x, spectrum_along_y


def plot_comparison(lr_data, hr_data, lr_mag, hr_mag,
                   aligned_lr_mag, aligned_hr_mag,
                   lr_mode_x, lr_mode_y, hr_mode_x, hr_mode_y,
                   domain_size, stats, output_file):
    """Create comprehensive comparison plot"""

    fig = plt.figure(figsize=(20, 12))
    Lx, Ly = domain_size

    # Row 1: Spatial maps
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(lr_data.T, origin='lower', cmap='viridis', aspect='auto',
                     extent=[0, Lx, 0, Ly])
    ax1.set_title(f'LR Perm Map\n{lr_data.shape[0]}×{lr_data.shape[1]} points')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(hr_data.T, origin='lower', cmap='viridis', aspect='auto',
                     extent=[0, Lx, 0, Ly])
    ax2.set_title(f'HR Perm Map\n{hr_data.shape[0]}×{hr_data.shape[1]} points')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    plt.colorbar(im2, ax=ax2)

    # Difference
    diff_mag = aligned_hr_mag - aligned_lr_mag

    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.imshow(diff_mag.T, origin='lower', cmap='RdBu_r', aspect='auto')
    ax3.set_title('FFT Magnitude Difference\n(HR - LR, aligned)')
    ax3.set_xlabel('kx mode')
    ax3.set_ylabel('ky mode')
    plt.colorbar(im3, ax=ax3)

    # Relative difference
    rel_diff = np.abs(diff_mag) / (aligned_lr_mag + 1e-20)

    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.imshow(np.log10(rel_diff.T + 1e-10), origin='lower',
                     cmap='hot', aspect='auto', vmin=-3, vmax=1)
    ax4.set_title('Relative Difference (log10)\n|HR-LR|/LR')
    ax4.set_xlabel('kx mode')
    ax4.set_ylabel('ky mode')
    plt.colorbar(im4, ax=ax4, label='log10(rel. diff)')

    # Row 2: 2D FFT magnitude
    ax5 = plt.subplot(3, 4, 5)
    im5 = ax5.imshow(np.log10(lr_mag.T + 1e-20), origin='lower', cmap='hot', aspect='auto')
    ax5.set_title(f'LR FFT Magnitude (log10)\nModes: {len(lr_mode_x)}×{len(lr_mode_y)}')
    ax5.set_xlabel('kx mode')
    ax5.set_ylabel('ky mode')
    plt.colorbar(im5, ax=ax5)

    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.imshow(np.log10(hr_mag.T + 1e-20), origin='lower', cmap='hot', aspect='auto')
    ax6.set_title(f'HR FFT Magnitude (log10)\nModes: {len(hr_mode_x)}×{len(hr_mode_y)}')
    ax6.set_xlabel('kx mode')
    ax6.set_ylabel('ky mode')
    plt.colorbar(im6, ax=ax6)

    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.imshow(np.log10(aligned_lr_mag.T + 1e-20), origin='lower', cmap='hot', aspect='auto')
    ax7.set_title('LR (aligned)')
    ax7.set_xlabel('kx mode')
    ax7.set_ylabel('ky mode')
    plt.colorbar(im7, ax=ax7)

    ax8 = plt.subplot(3, 4, 8)
    im8 = ax8.imshow(np.log10(aligned_hr_mag.T + 1e-20), origin='lower', cmap='hot', aspect='auto')
    ax8.set_title('HR (aligned)')
    ax8.set_xlabel('kx mode')
    ax8.set_ylabel('ky mode')
    plt.colorbar(im8, ax=ax8)

    # Row 3: 1D spectra along axes
    lr_spec_x, lr_spec_y = extract_1d_spectrum(aligned_lr_mag, lr_mode_x, lr_mode_y)
    hr_spec_x, hr_spec_y = extract_1d_spectrum(aligned_hr_mag, lr_mode_x, lr_mode_y)

    ax9 = plt.subplot(3, 4, 9)
    ax9.semilogy(lr_mode_x, lr_spec_x, 'o-', label='LR', markersize=5, linewidth=2)
    ax9.semilogy(lr_mode_x, hr_spec_x, 's--', label='HR', markersize=4, alpha=0.7)
    ax9.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax9.set_xlabel('kx mode (at ky=0)')
    ax9.set_ylabel('Magnitude')
    ax9.set_title('Aligned 1D Spectrum (X-axis)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    ax10 = plt.subplot(3, 4, 10)
    ax10.semilogy(lr_mode_y, lr_spec_y, 'o-', label='LR', markersize=5, linewidth=2)
    ax10.semilogy(lr_mode_y, hr_spec_y, 's--', label='HR', markersize=4, alpha=0.7)
    ax10.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax10.set_xlabel('ky mode (at kx=0)')
    ax10.set_ylabel('Magnitude')
    ax10.set_title('Aligned 1D Spectrum (Y-axis)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Differences in 1D
    diff_x = hr_spec_x - lr_spec_x
    diff_y = hr_spec_y - lr_spec_y

    ax11 = plt.subplot(3, 4, 11)
    ax11.plot(lr_mode_x, diff_x, 'o-', markersize=5, color='red')
    ax11.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax11.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax11.set_xlabel('kx mode')
    ax11.set_ylabel('HR - LR')
    ax11.set_title(f'Difference (X-axis)\nMax: {np.max(np.abs(diff_x)):.2e}')
    ax11.grid(True, alpha=0.3)

    ax12 = plt.subplot(3, 4, 12)
    ax12.plot(lr_mode_y, diff_y, 'o-', markersize=5, color='red')
    ax12.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax12.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax12.set_xlabel('ky mode')
    ax12.set_ylabel('HR - LR')
    ax12.set_title(f'Difference (Y-axis)\nMax: {np.max(np.abs(diff_y)):.2e}')
    ax12.grid(True, alpha=0.3)

    plt.suptitle(f'FFT Comparison (forward normalization + grid spacing)\n'
                 f'Max diff: {stats["max_diff"]:.2e}, Mean diff: {stats["mean_diff"]:.2e}, '
                 f'Mean rel diff: {stats["mean_rel_diff"]:.4f}',
                 fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def run_comparison(apply_log=False, epsilon=1e-20):
    """
    Run FFT comparison with optional log transformation

    Args:
        apply_log: If True, apply log transformation before FFT
        epsilon: Small value for log(x + epsilon)
    """
    transform_str = "log-transformed" if apply_log else "original"
    print("\n" + "=" * 80)
    print(f"FFT COMPARISON: LR vs HR ({transform_str}, forward normalization)")
    print("=" * 80)

    # Load data
    lr_file = "/home/geofluids/research/FNO/src/initial_perm/output_lr/perm_map_0.h5"
    hr_file = "/home/geofluids/research/FNO/src/initial_perm/output_hr/perm_map_0.h5"

    print(f"\nLoading maps (apply_log={apply_log}, epsilon={epsilon})...")
    lr_data, lr_disc, _ = load_perm_map(lr_file, apply_log=apply_log, epsilon=epsilon)
    hr_data, hr_disc, _ = load_perm_map(hr_file, apply_log=apply_log, epsilon=epsilon)

    print(f"  LR: {lr_data.shape}, discretization: {lr_disc}")
    print(f"  HR: {hr_data.shape}, discretization: {hr_disc}")

    # Calculate domain size
    lr_domain = (lr_data.shape[0] * lr_disc[0], lr_data.shape[1] * lr_disc[1])
    hr_domain = (hr_data.shape[0] * hr_disc[0], hr_data.shape[1] * hr_disc[1])

    print(f"\nPhysical domain:")
    print(f"  LR: {lr_domain[0]:.2f} m × {lr_domain[1]:.2f} m")
    print(f"  HR: {hr_domain[0]:.2f} m × {hr_domain[1]:.2f} m")

    # Compute FFT
    print("\nComputing FFT...")
    lr_mag, lr_mode_x, lr_mode_y = compute_fft_modes(lr_data, lr_domain)
    hr_mag, hr_mode_x, hr_mode_y = compute_fft_modes(hr_data, hr_domain)

    print(f"  LR modes: kx=[{lr_mode_x.min():.0f}, {lr_mode_x.max():.0f}], "
          f"ky=[{lr_mode_y.min():.0f}, {lr_mode_y.max():.0f}]")
    print(f"  HR modes: kx=[{hr_mode_x.min():.0f}, {hr_mode_x.max():.0f}], "
          f"ky=[{hr_mode_y.min():.0f}, {hr_mode_y.max():.0f}]")

    # Align spectra
    print("\nAligning spectra...")
    aligned_lr_mag, aligned_hr_mag = align_spectra(
        lr_mag, lr_mode_x, lr_mode_y, hr_mag, hr_mode_x, hr_mode_y)

    # Compute statistics
    diff = aligned_hr_mag - aligned_lr_mag
    rel_diff = np.abs(diff) / (aligned_lr_mag + 1e-20)

    stats = {
        'max_diff': np.max(np.abs(diff)),
        'mean_diff': np.mean(np.abs(diff)),
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff)
    }

    print("\nStatistics:")
    print(f"  Max absolute diff: {stats['max_diff']:.6e}")
    print(f"  Mean absolute diff: {stats['mean_diff']:.6e}")
    print(f"  Max relative diff: {stats['max_rel_diff']:.6f}")
    print(f"  Mean relative diff: {stats['mean_rel_diff']:.6f}")

    # Plot
    print("\nGenerating plot...")
    suffix = "_log" if apply_log else ""
    output_file = f"/home/geofluids/research/FNO/src/initial_perm/fft_comparison_forward{suffix}.png"
    plot_comparison(lr_data, hr_data, lr_mag, hr_mag,
                   aligned_lr_mag, aligned_hr_mag,
                   lr_mode_x, lr_mode_y, hr_mode_x, hr_mode_y,
                   hr_domain, stats, output_file)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()

    return stats


def main():
    """Main function - compare both original and log-transformed data"""

    print("\n" + "#" * 80)
    print("# FFT COMPARISON: Original vs Log-transformed")
    print("#" * 80)

    # Run comparison on original data
    print("\n" + "=" * 80)
    print("PART 1: Original permeability values")
    print("=" * 80)
    stats_original = run_comparison(apply_log=False)

    # Run comparison on log-transformed data
    print("\n" + "=" * 80)
    print("PART 2: Log-transformed permeability values")
    print("=" * 80)
    stats_log = run_comparison(apply_log=True, epsilon=1e-20)

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()
    print("Original data:")
    print(f"  Max abs diff: {stats_original['max_diff']:.6e}")
    print(f"  Mean abs diff: {stats_original['mean_diff']:.6e}")
    print(f"  Mean rel diff: {stats_original['mean_rel_diff']:.6f}")
    print()
    print("Log-transformed data:")
    print(f"  Max abs diff: {stats_log['max_diff']:.6e}")
    print(f"  Mean abs diff: {stats_log['mean_diff']:.6e}")
    print(f"  Mean rel diff: {stats_log['mean_rel_diff']:.6f}")
    print()
    print("Impact of log transformation:")
    print(f"  Max diff ratio (log/original): {stats_log['max_diff'] / stats_original['max_diff']:.2e}")
    print(f"  Mean diff ratio (log/original): {stats_log['mean_diff'] / stats_original['mean_diff']:.2e}")
    print(f"  Mean rel diff ratio (log/original): {stats_log['mean_rel_diff'] / stats_original['mean_rel_diff']:.2f}")
    print()


if __name__ == '__main__':
    main()
