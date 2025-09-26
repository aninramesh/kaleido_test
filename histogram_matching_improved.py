#!/usr/bin/env python3
"""
Improved Statistical Normalization for Pushbroom Spectral Bands

This enhanced version provides multiple normalization strategies to handle
bands with different dynamic ranges while minimizing clipping.
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

def calculate_robust_statistics(image_data, method='standard', percentile_range=(5, 95), verbose=False):
    """
    Calculate statistics using different methods to handle outliers and extreme values.

    Args:
        image_data: Image data array
        method: 'standard' (mean/std), 'percentile' (percentile-based), 'robust' (trimmed)
        percentile_range: Range for percentile-based statistics
        verbose: Enable verbose output
    """
    valid_mask = image_data > 0
    valid_pixels = image_data[valid_mask]

    if len(valid_pixels) == 0:
        return 0.0, 0.0, 0, {}

    stats_info = {}

    if method == 'standard':
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)
        stats_info = {
            'method': 'standard',
            'range': (valid_pixels.min(), valid_pixels.max()),
            'percentiles': (np.percentile(valid_pixels, 5), np.percentile(valid_pixels, 95))
        }

    elif method == 'percentile':
        # Use percentile-based approach for more robust statistics
        p_low, p_high = percentile_range
        p_low_val = np.percentile(valid_pixels, p_low)
        p_high_val = np.percentile(valid_pixels, p_high)

        # Use percentile range as "mean" and spread as "std"
        mean_val = (p_low_val + p_high_val) / 2
        std_val = (p_high_val - p_low_val) / 4  # Approximate std from range

        stats_info = {
            'method': 'percentile',
            'percentile_range': percentile_range,
            'percentile_values': (p_low_val, p_high_val),
            'range': (valid_pixels.min(), valid_pixels.max())
        }

    elif method == 'robust':
        # Trim extreme values (5% from each end) and calculate statistics
        p5 = np.percentile(valid_pixels, 5)
        p95 = np.percentile(valid_pixels, 95)
        trimmed_pixels = valid_pixels[(valid_pixels >= p5) & (valid_pixels <= p95)]

        mean_val = np.mean(trimmed_pixels)
        std_val = np.std(trimmed_pixels)

        stats_info = {
            'method': 'robust',
            'trimmed_range': (p5, p95),
            'range': (valid_pixels.min(), valid_pixels.max()),
            'trimmed_count': len(trimmed_pixels)
        }

    if verbose:
        print(f"    {method.title()} stats: mean={mean_val:.2f}, std={std_val:.2f}")
        print(f"    Valid pixels: {len(valid_pixels)}/{image_data.size} ({len(valid_pixels)/image_data.size*100:.1f}%)")
        if 'percentile_values' in stats_info:
            p_low_val, p_high_val = stats_info['percentile_values']
            print(f"    {percentile_range[0]}th-{percentile_range[1]}th percentile: {p_low_val:.1f} - {p_high_val:.1f}")

    return mean_val, std_val, len(valid_pixels), stats_info

def apply_adaptive_normalization(image_data, target_mean, target_std, source_mean, source_std,
                                method='adaptive', strength=1.0, preserve_range=True, verbose=False):
    """
    Apply normalization with different strategies to minimize clipping.

    Args:
        method: 'standard', 'adaptive', 'partial', 'range_preserving'
        strength: Normalization strength (0.0-1.0), 1.0 = full normalization
        preserve_range: Try to preserve the original data range
    """
    if verbose:
        print(f"    Applying {method} normalization (strength={strength:.2f})")
        print(f"    Target: mean={target_mean:.2f}, std={target_std:.2f}")
        print(f"    Source: mean={source_mean:.2f}, std={source_std:.2f}")

    if source_std == 0:
        print("    Warning: Source standard deviation is 0, returning original data")
        return image_data.copy(), {}

    valid_mask = image_data > 0
    result = image_data.copy().astype(np.float32)

    if method == 'standard':
        # Original formula
        result[valid_mask] = ((image_data[valid_mask] - source_mean) *
                             (target_std / source_std)) + target_mean

    elif method == 'adaptive':
        # Adaptive approach: adjust strength based on dynamic range compatibility
        source_range = np.percentile(image_data[valid_mask], 95) - np.percentile(image_data[valid_mask], 5)
        target_range = target_std * 4  # Approximate range from std

        # Reduce strength if source range is much smaller than target
        range_ratio = min(1.0, source_range / target_range)
        adaptive_strength = strength * range_ratio

        if verbose:
            print(f"    Range compatibility: {range_ratio:.3f}, adaptive strength: {adaptive_strength:.3f}")

        # Apply partial normalization
        normalized = ((image_data[valid_mask] - source_mean) * (target_std / source_std)) + target_mean
        result[valid_mask] = (image_data[valid_mask] * (1 - adaptive_strength) +
                             normalized * adaptive_strength)

    elif method == 'partial':
        # Partial normalization: blend original and normalized values
        normalized = ((image_data[valid_mask] - source_mean) * (target_std / source_std)) + target_mean
        result[valid_mask] = (image_data[valid_mask] * (1 - strength) + normalized * strength)

    elif method == 'range_preserving':
        # Preserve the original range while matching statistics
        source_min = np.percentile(image_data[valid_mask], 5)
        source_max = np.percentile(image_data[valid_mask], 95)
        source_range = source_max - source_min

        # Scale to match target std but preserve range
        scale_factor = min(1.0, target_std / source_std)

        # Normalize and then scale to fit in available range
        normalized = ((image_data[valid_mask] - source_mean) * scale_factor) + target_mean

        # Ensure it fits in 10-bit range
        normalized_min, normalized_max = normalized.min(), normalized.max()
        if normalized_max > 1023 or normalized_min < 0:
            # Rescale to fit
            available_range = 1023
            normalized_range = normalized_max - normalized_min
            if normalized_range > 0:
                scale_to_fit = min(1.0, available_range / normalized_range)
                normalized = ((normalized - normalized.mean()) * scale_to_fit) + target_mean

        result[valid_mask] = normalized

        if verbose:
            print(f"    Scale factor: {scale_factor:.3f}")

    # Clip to valid range
    result = np.clip(result, 0, 1023)
    result = result.astype(np.uint16)

    # Calculate clipping statistics
    original_normalized = ((image_data[valid_mask].astype(np.float32) - source_mean) *
                          (target_std / source_std)) + target_mean
    clipped_low = np.sum(original_normalized < 0)
    clipped_high = np.sum(original_normalized > 1023)

    clipping_info = {
        'clipped_low': clipped_low,
        'clipped_high': clipped_high,
        'total_valid': np.sum(valid_mask),
        'clipping_percent': (clipped_low + clipped_high) / np.sum(valid_mask) * 100
    }

    if verbose:
        result_mean, result_std, _, _ = calculate_robust_statistics(result, 'standard')
        print(f"    Result: mean={result_mean:.2f}, std={result_std:.2f}")
        if clipping_info['clipping_percent'] > 0:
            print(f"    Would have clipped: {clipping_info['clipping_percent']:.1f}% "
                  f"({clipped_low} low, {clipped_high} high)")

    return result, clipping_info

def auto_select_method_and_strength(source_stats, target_stats, verbose=False):
    """
    Automatically select the best normalization method and strength based on data characteristics.
    """
    source_mean, source_std = source_stats[:2]
    target_mean, target_std = target_stats[:2]

    # Calculate ratios to understand the transformation required
    mean_ratio = target_mean / source_mean if source_mean > 0 else 1.0
    std_ratio = target_std / source_std if source_std > 0 else 1.0

    # Determine if this is a challenging normalization
    challenging = (std_ratio > 3.0 or std_ratio < 0.5 or mean_ratio > 2.0 or mean_ratio < 0.5)

    if challenging:
        if std_ratio > 5.0:  # Very high stretch needed
            method = 'range_preserving'
            strength = 0.7
        elif std_ratio > 2.0:  # High stretch
            method = 'adaptive'
            strength = 0.8
        else:  # Moderate but challenging
            method = 'partial'
            strength = 0.9
    else:
        # Not challenging, use standard approach
        method = 'standard'
        strength = 1.0

    if verbose:
        print(f"    Auto-selected: {method} (strength={strength:.2f})")
        print(f"    Ratios - mean: {mean_ratio:.2f}, std: {std_ratio:.2f}, challenging: {challenging}")

    return method, strength

def save_10bit_tiff(img_array, output_path, verbose=False):
    """Save 10-bit image data as TIFF."""
    img_16bit = (img_array * 16).astype(np.uint16)
    Image.fromarray(img_16bit).save(output_path)
    if verbose:
        print(f"    Saved: {output_path}")

def create_enhanced_comparison_plot(band_statistics, output_filename="enhanced_normalization_stats.png"):
    """Create enhanced visualization with method information."""
    bands = [b for b in band_statistics.keys() if b != 'green_pan']
    if not bands:
        return

    # Prepare data
    original_means = [band_statistics[band]['original_mean'] for band in bands]
    original_stds = [band_statistics[band]['original_std'] for band in bands]
    normalized_means = [band_statistics[band]['normalized_mean'] for band in bands]
    normalized_stds = [band_statistics[band]['normalized_std'] for band in bands]
    methods = [band_statistics[band].get('method', 'standard') for band in bands]
    strengths = [band_statistics[band].get('strength', 1.0) for band in bands]

    ref_mean = band_statistics['green_pan']['original_mean']
    ref_std = band_statistics['green_pan']['original_std']

    # Create enhanced plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Statistical Normalization Results', fontsize=16, fontweight='bold')

    x = np.arange(len(bands))
    width = 0.35

    # Plot 1: Mean values
    bars1 = ax1.bar(x - width/2, original_means, width, label='Original', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, normalized_means, width, label='Normalized', alpha=0.8, color='lightgreen')
    ax1.axhline(y=ref_mean, color='red', linestyle='--', linewidth=2, label=f'Green/PAN target ({ref_mean:.0f})')

    ax1.set_xlabel('Spectral Bands')
    ax1.set_ylabel('Mean Pixel Value')
    ax1.set_title('Mean Values: Original vs Normalized')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (orig, norm) in enumerate(zip(original_means, normalized_means)):
        ax1.text(i - width/2, orig + max(original_means)*0.01, f'{orig:.0f}', ha='center', va='bottom')
        ax1.text(i + width/2, norm + max(normalized_means)*0.01, f'{norm:.0f}', ha='center', va='bottom')

    # Plot 2: Standard deviations
    bars3 = ax2.bar(x - width/2, original_stds, width, label='Original', alpha=0.8, color='skyblue')
    bars4 = ax2.bar(x + width/2, normalized_stds, width, label='Normalized', alpha=0.8, color='lightgreen')
    ax2.axhline(y=ref_std, color='red', linestyle='--', linewidth=2, label=f'Green/PAN target ({ref_std:.0f})')
    ax2.axhline(y=ref_std * 0.9, color='orange', linestyle=':', alpha=0.7, label='±10% tolerance')
    ax2.axhline(y=ref_std * 1.1, color='orange', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Spectral Bands')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Standard Deviation: Original vs Normalized')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for i, (orig, norm) in enumerate(zip(original_stds, normalized_stds)):
        ax2.text(i - width/2, orig + max(original_stds)*0.01, f'{orig:.0f}', ha='center', va='bottom')
        ax2.text(i + width/2, norm + max(normalized_stds)*0.01, f'{norm:.0f}', ha='center', va='bottom')

    # Plot 3: Methods used
    method_colors = {'standard': 'green', 'adaptive': 'blue', 'partial': 'orange', 'range_preserving': 'purple'}
    colors = [method_colors.get(m, 'gray') for m in methods]
    bars5 = ax3.bar(x, strengths, color=colors, alpha=0.7)

    ax3.set_xlabel('Spectral Bands')
    ax3.set_ylabel('Normalization Strength')
    ax3.set_title('Normalization Method and Strength Used')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bands, rotation=45)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)

    # Add method labels
    for i, (method, strength) in enumerate(zip(methods, strengths)):
        ax3.text(i, strength + 0.02, f'{method}\n{strength:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 4: Validation results
    validation_results = []
    for band in bands:
        norm_std = band_statistics[band]['normalized_std']
        within_tolerance = abs(norm_std - ref_std) <= ref_std * 0.1
        validation_results.append(1 if within_tolerance else 0)

    colors_val = ['green' if v else 'red' for v in validation_results]
    bars6 = ax4.bar(x, validation_results, color=colors_val, alpha=0.7)

    ax4.set_xlabel('Spectral Bands')
    ax4.set_ylabel('Validation Status')
    ax4.set_title('10% Tolerance Validation Results')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bands, rotation=45)
    ax4.set_ylim(0, 1.2)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['FAIL', 'PASS'])
    ax4.grid(True, alpha=0.3)

    # Add std deviation values
    for i, norm_std in enumerate(normalized_stds):
        status = 'PASS' if validation_results[i] else 'FAIL'
        ax4.text(i, validation_results[i] + 0.05, f'{norm_std:.0f}\n{status}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Enhanced comparison plot saved: {output_filename}")
    plt.show()

def main():
    """Main function with improved normalization strategies."""
    parser = argparse.ArgumentParser(description='Enhanced statistical normalization with multiple strategies')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_aligned_*_236images_start0_25pxsec_greenpan.tiff',
                       help='Pattern to match input files')
    parser.add_argument('--output-suffix', type=str, default='_enhanced_normalized',
                       help='Suffix for output files')
    parser.add_argument('--method', choices=['auto', 'standard', 'adaptive', 'partial', 'range_preserving'],
                       default='auto', help='Normalization method')
    parser.add_argument('--strength', type=float, default=1.0,
                       help='Normalization strength (0.0-1.0)')
    parser.add_argument('--stats-method', choices=['standard', 'percentile', 'robust'], default='standard',
                       help='Statistics calculation method')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Find input files
    current_dir = Path('.')
    band_files = {}
    expected_bands = ['green_pan', 'nir', 'blue', 'red_edge', 'red']

    print(f"Looking for files with pattern: {args.input_pattern}")
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

    print(f"\nProcessing {len(target_bands)} bands with enhanced normalization")
    print(f"Method: {args.method}, Strength: {args.strength}, Stats: {args.stats_method}")

    # Calculate reference statistics
    print("\n1. Calculating reference statistics (Green/PAN)...")
    ref_image = read_geotiff_10bit(band_files['green_pan'])
    ref_mean, ref_std, ref_count, ref_info = calculate_robust_statistics(
        ref_image, args.stats_method, verbose=args.verbose)
    print(f"   Green/PAN: mean={ref_mean:.2f}, std={ref_std:.2f}")

    # Store results
    band_statistics = {
        'green_pan': {
            'original_mean': ref_mean, 'original_std': ref_std,
            'normalized_mean': ref_mean, 'normalized_std': ref_std
        }
    }

    # Process each target band
    print("\n2. Processing target bands...")
    validation_results = {}

    for band_name in target_bands:
        print(f"\n  Processing {band_name} band...")

        # Read and analyze
        target_image = read_geotiff_10bit(band_files[band_name])
        orig_mean, orig_std, orig_count, orig_info = calculate_robust_statistics(
            target_image, args.stats_method, verbose=args.verbose)
        print(f"    Original: mean={orig_mean:.2f}, std={orig_std:.2f}")

        # Select method and strength
        if args.method == 'auto':
            method, strength = auto_select_method_and_strength(
                (orig_mean, orig_std), (ref_mean, ref_std), args.verbose)
        else:
            method, strength = args.method, args.strength

        # Apply normalization
        normalized_image, clipping_info = apply_adaptive_normalization(
            target_image, ref_mean, ref_std, orig_mean, orig_std,
            method=method, strength=strength, verbose=args.verbose)

        # Validate results
        norm_mean, norm_std, norm_count, _ = calculate_robust_statistics(normalized_image, 'standard')
        within_tolerance = abs(norm_std - ref_std) <= ref_std * 0.1

        print(f"    Result: mean={norm_mean:.2f}, std={norm_std:.2f}")
        print(f"    Validation: {'PASS' if within_tolerance else 'FAIL'} "
              f"(target: {ref_std:.2f} ±10%)")

        validation_results[band_name] = within_tolerance

        # Store statistics
        band_statistics[band_name] = {
            'original_mean': orig_mean, 'original_std': orig_std,
            'normalized_mean': norm_mean, 'normalized_std': norm_std,
            'method': method, 'strength': strength,
            'clipping_percent': clipping_info['clipping_percent']
        }

        # Save result
        input_path = band_files[band_name]
        output_path = input_path.stem + args.output_suffix + input_path.suffix
        save_10bit_tiff(normalized_image, output_path, args.verbose)
        print(f"    Saved: {output_path}")

    # Final summary
    passed = sum(validation_results.values())
    total = len(validation_results)
    print(f"\n3. Enhanced Normalization Summary:")
    print(f"   Validation: {passed}/{total} bands within 10% tolerance")
    print(f"   Methods used: {set(band_statistics[b].get('method', 'standard') for b in target_bands)}")

    # Create enhanced visualization
    print("\n4. Creating enhanced comparison plot...")
    create_enhanced_comparison_plot(band_statistics)

if __name__ == "__main__":
    main()