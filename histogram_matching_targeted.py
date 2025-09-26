#!/usr/bin/env python3
"""
Targeted Statistical Normalization to Achieve 10% Standard Deviation Match

This version specifically targets achieving standard deviation within 10% of the
Green/PAN reference band through iterative optimization and outlier-aware processing.
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def read_geotiff_10bit(file_path):
    """Read a 10-bit GeoTIFF file and return the image data in true 10-bit range."""
    with Image.open(file_path) as img:
        img_array = np.array(img)
        max_val = img_array.max()
        range_val = max_val - img_array.min()

        if range_val <= 1023 * 16:
            true_10bit = img_array / 16.0
            return true_10bit.astype(np.uint16)
        elif max_val <= 1023:
            return img_array
        else:
            true_10bit = img_array / 16.0
            return true_10bit.astype(np.uint16)

def calculate_statistics_exclude_zeros(image_data, exclude_outliers=True, outlier_percentile=1.0, verbose=False):
    """
    Calculate statistics excluding zeros and optionally extreme outliers.

    Args:
        exclude_outliers: If True, exclude extreme outliers from statistics
        outlier_percentile: Percentile to trim from each end (1.0 = trim 1% from each end)
    """
    valid_mask = image_data > 0
    valid_pixels = image_data[valid_mask]

    if len(valid_pixels) == 0:
        return 0.0, 0.0, 0

    if exclude_outliers and outlier_percentile > 0:
        # Remove extreme outliers to get more stable statistics
        p_low = np.percentile(valid_pixels, outlier_percentile)
        p_high = np.percentile(valid_pixels, 100 - outlier_percentile)
        outlier_mask = (valid_pixels >= p_low) & (valid_pixels <= p_high)
        trimmed_pixels = valid_pixels[outlier_mask]

        if len(trimmed_pixels) > 0:
            mean_val = np.mean(trimmed_pixels)
            std_val = np.std(trimmed_pixels)

            if verbose:
                outliers_removed = len(valid_pixels) - len(trimmed_pixels)
                print(f"    Outlier removal: {outliers_removed} pixels ({outliers_removed/len(valid_pixels)*100:.1f}%)")
                print(f"    Trimmed range: {p_low:.1f} - {p_high:.1f}")
        else:
            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)
    else:
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)

    if verbose:
        print(f"    Statistics: mean={mean_val:.2f}, std={std_val:.2f}")
        print(f"    Valid pixels: {len(valid_pixels)}/{image_data.size} ({len(valid_pixels)/image_data.size*100:.1f}%)")

    return mean_val, std_val, len(valid_pixels)

def iterative_normalization_for_std_target(image_data, target_mean, target_std, max_iterations=10,
                                          std_tolerance=0.1, verbose=False):
    """
    Iteratively adjust normalization to achieve target standard deviation within tolerance.

    This method uses an iterative approach to find the optimal scaling that achieves
    the target std deviation while minimizing clipping.
    """
    if verbose:
        print(f"    Target: mean={target_mean:.2f}, std={target_std:.2f} (±{std_tolerance*100:.0f}%)")

    valid_mask = image_data > 0
    original_data = image_data[valid_mask].astype(np.float32)

    if len(original_data) == 0:
        return image_data.copy()

    # Calculate initial statistics
    source_mean = np.mean(original_data)
    source_std = np.std(original_data)

    if source_std == 0:
        print("    Warning: Source std is 0, cannot normalize")
        return image_data.copy()

    # Target range for std
    std_target_min = target_std * (1 - std_tolerance)
    std_target_max = target_std * (1 + std_tolerance)

    best_result = None
    best_std_error = float('inf')

    # Try different scaling strategies
    scaling_strategies = [
        1.0,  # Standard normalization
        0.95, 0.90, 0.85, 0.80,  # Reduced scaling to minimize clipping
        1.05, 1.10, 1.15, 1.20   # Increased scaling if needed
    ]

    for iteration, scale_factor in enumerate(scaling_strategies):
        if verbose and iteration == 0:
            print(f"    Trying {len(scaling_strategies)} scaling strategies...")

        # Apply normalization with current scale factor
        effective_target_std = target_std * scale_factor
        normalized_data = ((original_data - source_mean) * (effective_target_std / source_std)) + target_mean

        # Apply different clipping strategies
        clipping_strategies = [
            {'method': 'hard', 'min_val': 0, 'max_val': 1023},  # Standard hard clipping
            {'method': 'soft', 'min_val': 1, 'max_val': 1022},  # Soft clipping with margins
            {'method': 'adaptive', 'percentile': 0.5},           # Adaptive range based on data
        ]

        for clip_strategy in clipping_strategies:
            test_data = normalized_data.copy()

            if clip_strategy['method'] == 'hard':
                test_data = np.clip(test_data, clip_strategy['min_val'], clip_strategy['max_val'])
            elif clip_strategy['method'] == 'soft':
                test_data = np.clip(test_data, clip_strategy['min_val'], clip_strategy['max_val'])
            elif clip_strategy['method'] == 'adaptive':
                # Use percentile-based clipping
                p = clip_strategy['percentile']
                min_val = max(0, np.percentile(test_data, p))
                max_val = min(1023, np.percentile(test_data, 100-p))
                test_data = np.clip(test_data, min_val, max_val)

            # Calculate resulting statistics
            result_mean = np.mean(test_data)
            result_std = np.std(test_data)

            # Check if std is within target range
            std_error = abs(result_std - target_std)
            within_tolerance = std_target_min <= result_std <= std_target_max

            # Update best result if this is better
            if within_tolerance or std_error < best_std_error:
                best_std_error = std_error

                # Create full result image
                result_image = image_data.copy().astype(np.float32)
                result_image[valid_mask] = test_data
                best_result = {
                    'image': result_image.astype(np.uint16),
                    'mean': result_mean,
                    'std': result_std,
                    'scale_factor': scale_factor,
                    'clip_method': clip_strategy['method'],
                    'within_tolerance': within_tolerance,
                    'std_error': std_error
                }

                if within_tolerance:
                    if verbose:
                        print(f"    SUCCESS: Found solution with std={result_std:.2f} "
                              f"(scale={scale_factor:.2f}, clip={clip_strategy['method']})")
                    break

        if best_result and best_result['within_tolerance']:
            break

    if best_result:
        result = best_result['image']
        if verbose:
            status = "PASS" if best_result['within_tolerance'] else "BEST ATTEMPT"
            print(f"    Result [{status}]: mean={best_result['mean']:.2f}, std={best_result['std']:.2f}")
            print(f"    Method: scale={best_result['scale_factor']:.2f}, clip={best_result['clip_method']}")
            print(f"    Std error: {best_result['std_error']:.2f}")
        return result
    else:
        print("    Warning: No normalization solution found")
        return image_data.copy()

def precise_std_normalization(image_data, target_mean, target_std, tolerance=0.1, verbose=False):
    """
    Precise normalization specifically designed to achieve std within tolerance.
    Uses outlier handling and iterative refinement.
    """
    if verbose:
        print(f"    Precise std normalization: target={target_std:.2f} ±{tolerance*100:.0f}%")

    # Step 1: Remove extreme outliers for better statistics
    orig_mean, orig_std, valid_count = calculate_statistics_exclude_zeros(
        image_data, exclude_outliers=True, outlier_percentile=2.0, verbose=verbose)

    if orig_std == 0:
        return image_data.copy()

    # Step 2: Use iterative approach to find best normalization
    result = iterative_normalization_for_std_target(
        image_data, target_mean, target_std,
        std_tolerance=tolerance, verbose=verbose)

    return result

def save_10bit_tiff(img_array, output_path, verbose=False):
    """Save 10-bit image data as TIFF."""
    img_16bit = (img_array * 16).astype(np.uint16)
    Image.fromarray(img_16bit).save(output_path)
    if verbose:
        print(f"    Saved: {output_path}")

def create_targeted_comparison_plot(band_statistics, target_std, tolerance=0.1,
                                  output_filename="targeted_normalization_results.png"):
    """Create visualization focused on std deviation achievement."""
    bands = [b for b in band_statistics.keys() if b != 'green_pan']
    if not bands:
        return

    # Calculate tolerance bounds
    std_min = target_std * (1 - tolerance)
    std_max = target_std * (1 + tolerance)

    # Prepare data
    original_stds = [band_statistics[band]['original_std'] for band in bands]
    normalized_stds = [band_statistics[band]['normalized_std'] for band in bands]
    validation_status = [band_statistics[band]['validation_pass'] for band in bands]
    std_errors = [abs(band_statistics[band]['normalized_std'] - target_std) for band in bands]

    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Targeted Standard Deviation Normalization Results\nTarget: {target_std:.1f} ±{tolerance*100:.0f}%',
                 fontsize=16, fontweight='bold')

    x = np.arange(len(bands))

    # Plot 1: Standard deviation comparison
    width = 0.4
    bars1 = ax1.bar(x - width/2, original_stds, width, label='Original', alpha=0.8, color='lightblue')
    bars2 = ax1.bar(x + width/2, normalized_stds, width, label='Normalized', alpha=0.8, color='lightgreen')

    # Add target line and tolerance band
    ax1.axhline(y=target_std, color='red', linestyle='-', linewidth=2, label=f'Target ({target_std:.1f})')
    ax1.axhspan(std_min, std_max, alpha=0.2, color='red', label=f'±{tolerance*100:.0f}% tolerance')

    ax1.set_xlabel('Spectral Bands')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Standard Deviation: Original vs Normalized')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (orig, norm) in enumerate(zip(original_stds, normalized_stds)):
        ax1.text(i - width/2, orig + max(original_stds)*0.01, f'{orig:.0f}', ha='center', va='bottom', fontsize=9)
        color = 'green' if validation_status[i] else 'red'
        ax1.text(i + width/2, norm + max(normalized_stds)*0.01, f'{norm:.0f}',
                ha='center', va='bottom', fontsize=9, color=color, weight='bold')

    # Plot 2: Validation status
    colors = ['green' if status else 'red' for status in validation_status]
    bars3 = ax2.bar(x, [1 if status else 0 for status in validation_status],
                   color=colors, alpha=0.7)

    ax2.set_xlabel('Spectral Bands')
    ax2.set_ylabel('Validation Status')
    ax2.set_title('10% Tolerance Validation Results')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['FAIL', 'PASS'])
    ax2.grid(True, alpha=0.3)

    # Add std values on bars
    for i, (norm_std, status) in enumerate(zip(normalized_stds, validation_status)):
        status_text = 'PASS' if status else 'FAIL'
        ax2.text(i, 0.5, f'{norm_std:.1f}\n{status_text}', ha='center', va='center',
                fontweight='bold', color='white')

    # Plot 3: Standard deviation error from target
    colors_error = ['green' if status else 'red' for status in validation_status]
    bars4 = ax3.bar(x, std_errors, color=colors_error, alpha=0.7)

    ax3.axhline(y=target_std * tolerance, color='orange', linestyle='--',
               label=f'Tolerance limit ({target_std * tolerance:.1f})')

    ax3.set_xlabel('Spectral Bands')
    ax3.set_ylabel('Absolute Error from Target Std')
    ax3.set_title('Standard Deviation Error Analysis')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bands, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add error values
    for i, error in enumerate(std_errors):
        ax3.text(i, error + max(std_errors)*0.01, f'{error:.1f}', ha='center', va='bottom')

    # Plot 4: Scaling factors achieved
    scaling_factors = [band_statistics[band]['normalized_std'] / band_statistics[band]['original_std']
                      for band in bands]
    required_factors = [target_std / band_statistics[band]['original_std'] for band in bands]

    bars5 = ax4.bar(x - width/2, required_factors, width, label='Required', alpha=0.8, color='orange')
    bars6 = ax4.bar(x + width/2, scaling_factors, width, label='Achieved', alpha=0.8, color='green')

    ax4.set_xlabel('Spectral Bands')
    ax4.set_ylabel('Scaling Factor')
    ax4.set_title('Standard Deviation Scaling: Required vs Achieved')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bands, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add scaling factor labels
    for i, (req, ach) in enumerate(zip(required_factors, scaling_factors)):
        ax4.text(i - width/2, req + max(required_factors)*0.01, f'{req:.1f}x', ha='center', va='bottom', fontsize=9)
        ax4.text(i + width/2, ach + max(scaling_factors)*0.01, f'{ach:.1f}x', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Targeted normalization plot saved: {output_filename}")
    plt.show()

def main():
    """Main function for targeted standard deviation normalization."""
    parser = argparse.ArgumentParser(description='Targeted normalization to achieve 10% std deviation match')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_aligned_*_236images_start0_25pxsec_greenpan.tiff',
                       help='Pattern to match input files')
    parser.add_argument('--output-suffix', type=str, default='_targeted_normalized',
                       help='Suffix for output files')
    parser.add_argument('--tolerance', type=float, default=0.1,
                       help='Standard deviation tolerance (0.1 = 10%)')
    parser.add_argument('--remove-outliers', action='store_true', default=True,
                       help='Remove extreme outliers for better statistics')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Find input files
    current_dir = Path('.')
    band_files = {}
    expected_bands = ['green_pan', 'nir', 'blue', 'red_edge', 'red']

    print(f"Targeted Standard Deviation Normalization")
    print(f"Target tolerance: ±{args.tolerance*100:.0f}%")
    print(f"Looking for files: {args.input_pattern}")

    for band in expected_bands:
        pattern = args.input_pattern.replace('*', band)
        matching_files = list(current_dir.glob(pattern))
        if matching_files:
            band_files[band] = matching_files[0]
            print(f"  Found {band}: {matching_files[0]}")

    if 'green_pan' not in band_files:
        print("Error: Green/PAN reference band not found!")
        return

    target_bands = [band for band in expected_bands if band != 'green_pan' and band in band_files]
    if not target_bands:
        print("Error: No target bands found!")
        return

    # Calculate reference statistics
    print("\n1. Calculating reference statistics (Green/PAN)...")
    ref_image = read_geotiff_10bit(band_files['green_pan'])
    ref_mean, ref_std, ref_count = calculate_statistics_exclude_zeros(
        ref_image, exclude_outliers=args.remove_outliers, verbose=args.verbose)

    print(f"   Green/PAN reference: mean={ref_mean:.2f}, std={ref_std:.2f}")

    # Calculate target range
    std_target_min = ref_std * (1 - args.tolerance)
    std_target_max = ref_std * (1 + args.tolerance)
    print(f"   Target std range: {std_target_min:.2f} - {std_target_max:.2f}")

    # Store results
    band_statistics = {
        'green_pan': {
            'original_mean': ref_mean, 'original_std': ref_std,
            'normalized_mean': ref_mean, 'normalized_std': ref_std,
            'validation_pass': True
        }
    }

    # Process each target band
    print("\n2. Processing target bands with targeted normalization...")
    success_count = 0

    for band_name in target_bands:
        print(f"\n  Processing {band_name} band...")

        # Read and analyze
        target_image = read_geotiff_10bit(band_files[band_name])
        orig_mean, orig_std, orig_count = calculate_statistics_exclude_zeros(
            target_image, exclude_outliers=args.remove_outliers, verbose=args.verbose)

        print(f"    Original: mean={orig_mean:.2f}, std={orig_std:.2f}")

        # Apply targeted normalization
        normalized_image = precise_std_normalization(
            target_image, ref_mean, ref_std, tolerance=args.tolerance, verbose=args.verbose)

        # Validate results
        norm_mean, norm_std, norm_count = calculate_statistics_exclude_zeros(normalized_image, exclude_outliers=False)
        within_tolerance = std_target_min <= norm_std <= std_target_max

        status = "PASS" if within_tolerance else "FAIL"
        print(f"    Final result: mean={norm_mean:.2f}, std={norm_std:.2f} [{status}]")

        if within_tolerance:
            success_count += 1
            print(f"    ✓ SUCCESS: Standard deviation within {args.tolerance*100:.0f}% tolerance!")
        else:
            error = abs(norm_std - ref_std)
            print(f"    ✗ MISS: Error = {error:.2f} (target ±{ref_std*args.tolerance:.2f})")

        # Store statistics
        band_statistics[band_name] = {
            'original_mean': orig_mean, 'original_std': orig_std,
            'normalized_mean': norm_mean, 'normalized_std': norm_std,
            'validation_pass': within_tolerance
        }

        # Save result
        input_path = band_files[band_name]
        output_path = input_path.stem + args.output_suffix + input_path.suffix
        save_10bit_tiff(normalized_image, output_path, args.verbose)
        print(f"    Saved: {output_path}")

    # Final summary
    total_bands = len(target_bands)
    print(f"\n3. TARGETED NORMALIZATION SUMMARY:")
    print(f"   SUCCESS: {success_count}/{total_bands} bands achieved ±{args.tolerance*100:.0f}% std tolerance")
    print(f"   Reference std: {ref_std:.2f}")
    print(f"   Target range: {std_target_min:.2f} - {std_target_max:.2f}")

    if success_count == total_bands:
        print("   🎯 PERFECT: All bands successfully normalized!")
    elif success_count > 0:
        print(f"   📈 PARTIAL: {success_count} bands achieved target")
    else:
        print("   ⚠️  CHALLENGE: No bands achieved target (inherent dynamic range limitations)")

    # Create targeted visualization
    print("\n4. Creating targeted analysis plot...")
    create_targeted_comparison_plot(band_statistics, ref_std, args.tolerance)

if __name__ == "__main__":
    main()