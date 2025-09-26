#!/usr/bin/env python3
"""
Statistical Normalization (Histogram Matching) for Pushbroom Spectral Bands

This script performs statistical normalization to adjust pixel value distribution
of NIR, Blue, Red Edge, and Red bands to match the Green/PAN reference band.

The normalization adjusts each band's mean and standard deviation to match
those of the Green reference band, standardizing radiometric quality across
all spectral bands.
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def read_geotiff_10bit(file_path):
    """
    Read a 10-bit GeoTIFF file and return the image data in true 10-bit range.

    Args:
        file_path (str): Path to the GeoTIFF file

    Returns:
        numpy.ndarray: Image data in true 10-bit range (0-1023)
    """
    with Image.open(file_path) as img:
        img_array = np.array(img)

        # Check if it's 10-bit data scaled by 16 (common in GeoTIFF)
        max_val = img_array.max()
        min_val = img_array.min()
        range_val = max_val - min_val

        if range_val <= 1023 * 16:  # 10-bit scaled by 16
            # Convert back to true 10-bit range (0-1023)
            true_10bit = img_array / 16.0
            return true_10bit.astype(np.uint16)
        elif max_val <= 1023:
            # True 10-bit data
            return img_array
        else:
            print(f"Warning: Data doesn't appear to be 10-bit format")
            true_10bit = img_array / 16.0
            return true_10bit.astype(np.uint16)

def calculate_statistics_exclude_zeros(image_data, verbose=False):
    """
    Calculate mean and standard deviation of image data, excluding pixels with value 0.

    Args:
        image_data (numpy.ndarray): Image data
        verbose (bool): Enable verbose output

    Returns:
        tuple: (mean, std_deviation, valid_pixel_count)
    """
    # Create mask for non-zero pixels
    valid_mask = image_data > 0
    valid_pixels = image_data[valid_mask]

    if len(valid_pixels) == 0:
        print("Warning: No valid (non-zero) pixels found in image")
        return 0.0, 0.0, 0

    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)

    if verbose:
        total_pixels = image_data.size
        valid_count = len(valid_pixels)
        print(f"    Valid pixels: {valid_count}/{total_pixels} ({valid_count/total_pixels*100:.1f}%)")
        print(f"    Mean: {mean_val:.2f}, Std: {std_val:.2f}")

    return mean_val, std_val, len(valid_pixels)

def apply_normalization_formula(image_data, target_mean, target_std, source_mean, source_std, verbose=False):
    """
    Apply statistical normalization formula to transform image data.

    Formula: P_new = ((P_original - μ_source) × (σ_target / σ_source)) + μ_target

    Args:
        image_data (numpy.ndarray): Original image data
        target_mean (float): Target mean (from reference band)
        target_std (float): Target standard deviation (from reference band)
        source_mean (float): Source mean (from current band)
        source_std (float): Source standard deviation (from current band)
        verbose (bool): Enable verbose output

    Returns:
        numpy.ndarray: Normalized image data with values clipped to [0, 1023]
    """
    if verbose:
        print(f"    Applying normalization formula:")
        print(f"    Target: mean={target_mean:.2f}, std={target_std:.2f}")
        print(f"    Source: mean={source_mean:.2f}, std={source_std:.2f}")

    # Avoid division by zero
    if source_std == 0:
        print("    Warning: Source standard deviation is 0, returning original data")
        return image_data.copy()

    # Apply normalization formula
    # P_new = ((P_original - μ_source) × (σ_target / σ_source)) + μ_target
    normalized_data = ((image_data.astype(np.float32) - source_mean) *
                      (target_std / source_std)) + target_mean

    # Clip values to valid 10-bit range [0, 1023]
    clipped_data = np.clip(normalized_data, 0, 1023)

    # Convert back to uint16 to maintain consistency
    result = clipped_data.astype(np.uint16)

    if verbose:
        # Calculate statistics on the result (excluding zeros for consistency)
        result_mean, result_std, _ = calculate_statistics_exclude_zeros(result)
        print(f"    Result: mean={result_mean:.2f}, std={result_std:.2f}")

        # Check how many values were clipped
        clipped_low = np.sum(normalized_data < 0)
        clipped_high = np.sum(normalized_data > 1023)
        total_pixels = normalized_data.size
        if clipped_low > 0 or clipped_high > 0:
            print(f"    Clipped: {clipped_low} low values, {clipped_high} high values "
                  f"({(clipped_low + clipped_high)/total_pixels*100:.2f}% total)")

    return result

def validate_normalization_within_tolerance(normalized_std, target_std, tolerance=0.1, band_name="", verbose=False):
    """
    Validate that normalized band standard deviation is within specified tolerance of target.

    Args:
        normalized_std (float): Standard deviation of normalized band
        target_std (float): Target standard deviation (reference band)
        tolerance (float): Tolerance as fraction (0.1 = 10%)
        band_name (str): Name of the band for reporting
        verbose (bool): Enable verbose output

    Returns:
        bool: True if within tolerance, False otherwise
    """
    lower_bound = target_std * (1 - tolerance)
    upper_bound = target_std * (1 + tolerance)
    within_tolerance = lower_bound <= normalized_std <= upper_bound

    if verbose or not within_tolerance:
        status = "PASS" if within_tolerance else "FAIL"
        print(f"    {band_name} validation [{status}]: std={normalized_std:.2f}, "
              f"target={target_std:.2f} ±{tolerance*100:.0f}% "
              f"[{lower_bound:.2f}, {upper_bound:.2f}]")

    return within_tolerance

def save_10bit_tiff(img_array, output_path, verbose=False):
    """
    Save 10-bit image data as TIFF preserving the 10-bit range.

    Args:
        img_array (numpy.ndarray): 10-bit image data (0-1023)
        output_path (str): Output file path
        verbose (bool): Enable verbose output
    """
    # Save as 16-bit TIFF to preserve 10-bit precision
    # Scale back to 16-bit range for storage (multiply by 16)
    img_16bit = (img_array * 16).astype(np.uint16)

    # Save as TIFF
    Image.fromarray(img_16bit).save(output_path)

    if verbose:
        print(f"    Saved: {output_path}")
        print(f"    Output shape: {img_array.shape}")
        print(f"    Output range: {img_array.min()}-{img_array.max()}")

def create_statistics_comparison_plot(band_statistics, output_filename="normalization_statistics.png"):
    """
    Create a visualization comparing statistics before and after normalization.

    Args:
        band_statistics (dict): Dictionary with band statistics
        output_filename (str): Output filename for the plot
    """
    bands = list(band_statistics.keys())
    if 'green_pan' in bands:
        bands.remove('green_pan')  # Remove reference band from comparison

    if not bands:
        print("No bands to plot")
        return

    # Prepare data for plotting
    original_means = [band_statistics[band]['original_mean'] for band in bands]
    original_stds = [band_statistics[band]['original_std'] for band in bands]
    normalized_means = [band_statistics[band]['normalized_mean'] for band in bands]
    normalized_stds = [band_statistics[band]['normalized_std'] for band in bands]

    # Reference values (Green/PAN)
    ref_mean = band_statistics['green_pan']['original_mean']
    ref_std = band_statistics['green_pan']['original_std']

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(bands))
    width = 0.35

    # Plot means
    ax1.bar(x - width/2, original_means, width, label='Original', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, normalized_means, width, label='Normalized', alpha=0.8, color='lightgreen')
    ax1.axhline(y=ref_mean, color='red', linestyle='--', linewidth=2, label=f'Green/PAN target ({ref_mean:.1f})')

    ax1.set_xlabel('Spectral Bands')
    ax1.set_ylabel('Mean Pixel Value')
    ax1.set_title('Mean Values: Original vs Normalized')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (orig, norm) in enumerate(zip(original_means, normalized_means)):
        ax1.text(i - width/2, orig + max(original_means)*0.01, f'{orig:.1f}',
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, norm + max(normalized_means)*0.01, f'{norm:.1f}',
                ha='center', va='bottom', fontsize=9)

    # Plot standard deviations
    ax2.bar(x - width/2, original_stds, width, label='Original', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, normalized_stds, width, label='Normalized', alpha=0.8, color='lightgreen')
    ax2.axhline(y=ref_std, color='red', linestyle='--', linewidth=2, label=f'Green/PAN target ({ref_std:.1f})')

    # Add 10% tolerance bands
    ax2.axhline(y=ref_std * 0.9, color='orange', linestyle=':', alpha=0.7, label='±10% tolerance')
    ax2.axhline(y=ref_std * 1.1, color='orange', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Spectral Bands')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Standard Deviation: Original vs Normalized')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (orig, norm) in enumerate(zip(original_stds, normalized_stds)):
        ax2.text(i - width/2, orig + max(original_stds)*0.01, f'{orig:.1f}',
                ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, norm + max(normalized_stds)*0.01, f'{norm:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Statistics comparison plot saved: {output_filename}")
    plt.show()

def main():
    """Main function to perform statistical normalization on pushbroom bands."""

    parser = argparse.ArgumentParser(description='Perform statistical normalization (histogram matching) on pushbroom spectral bands')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_aligned_*_greenpan.tiff',
                       help='Pattern to match input mosaicked files (default: pushbroom_aligned_*_greenpan.tiff)')
    parser.add_argument('--output-suffix', type=str, default='_normalized',
                       help='Suffix to add to output normalized files (default: _normalized)')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                       help='Enable verbose output (default: False)')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Create statistics comparison plot (default: True)')
    parser.add_argument('--tolerance', type=float, default=0.1,
                       help='Validation tolerance as fraction (0.1 = 10%%, default: 0.1)')

    args = parser.parse_args()

    # Find mosaicked input files
    input_pattern = args.input_pattern
    current_dir = Path('.')

    # Define band mappings
    band_files = {}
    expected_bands = ['green_pan', 'nir', 'blue', 'red_edge', 'red']

    print(f"Looking for mosaicked files with pattern: {input_pattern}")

    # Find files for each band
    for band in expected_bands:
        # Look for files containing the band name
        pattern = input_pattern.replace('*', band)
        matching_files = list(current_dir.glob(pattern))

        if matching_files:
            band_files[band] = matching_files[0]
            print(f"  Found {band}: {matching_files[0]}")
        else:
            print(f"  Warning: No file found for {band} band with pattern {pattern}")

    # Check if we have the required reference band (Green/PAN)
    if 'green_pan' not in band_files:
        print("Error: Green/PAN reference band file not found!")
        return

    # Check if we have at least one target band to normalize
    target_bands = [band for band in expected_bands if band != 'green_pan' and band in band_files]
    if not target_bands:
        print("Error: No target bands found to normalize!")
        return

    print(f"\nProcessing {len(target_bands)} target bands against Green/PAN reference")
    print(f"Target bands: {', '.join(target_bands)}")

    # Step 1: Calculate statistics for the reference band (Green/PAN)
    print(f"\n1. Calculating reference statistics for Green/PAN band...")
    ref_image = read_geotiff_10bit(band_files['green_pan'])
    if ref_image is None:
        print("Error: Failed to read Green/PAN reference image")
        return

    ref_mean, ref_std, ref_valid_count = calculate_statistics_exclude_zeros(ref_image, args.verbose)
    print(f"   Green/PAN reference: mean={ref_mean:.2f}, std={ref_std:.2f}, valid_pixels={ref_valid_count}")

    # Store statistics for plotting
    band_statistics = {
        'green_pan': {
            'original_mean': ref_mean,
            'original_std': ref_std,
            'normalized_mean': ref_mean,  # Reference doesn't change
            'normalized_std': ref_std
        }
    }

    # Step 2: Process each target band
    print(f"\n2. Processing target bands...")
    validation_results = {}

    for band_name in target_bands:
        print(f"\n  Processing {band_name} band...")

        # Read target band image
        target_image = read_geotiff_10bit(band_files[band_name])
        if target_image is None:
            print(f"    Error: Failed to read {band_name} image")
            continue

        # Calculate original statistics
        orig_mean, orig_std, orig_valid_count = calculate_statistics_exclude_zeros(target_image, args.verbose)
        print(f"    Original {band_name}: mean={orig_mean:.2f}, std={orig_std:.2f}, valid_pixels={orig_valid_count}")

        # Apply normalization
        normalized_image = apply_normalization_formula(
            target_image, ref_mean, ref_std, orig_mean, orig_std, args.verbose
        )

        # Calculate normalized statistics for validation
        norm_mean, norm_std, norm_valid_count = calculate_statistics_exclude_zeros(normalized_image, args.verbose)
        print(f"    Normalized {band_name}: mean={norm_mean:.2f}, std={norm_std:.2f}, valid_pixels={norm_valid_count}")

        # Validate normalization
        is_valid = validate_normalization_within_tolerance(
            norm_std, ref_std, args.tolerance, band_name, args.verbose
        )
        validation_results[band_name] = is_valid

        # Store statistics for plotting
        band_statistics[band_name] = {
            'original_mean': orig_mean,
            'original_std': orig_std,
            'normalized_mean': norm_mean,
            'normalized_std': norm_std
        }

        # Save normalized image
        input_path = band_files[band_name]
        output_path = input_path.stem + args.output_suffix + input_path.suffix
        save_10bit_tiff(normalized_image, output_path, args.verbose)
        print(f"    Saved normalized {band_name}: {output_path}")

    # Step 3: Summary and validation report
    print(f"\n3. Normalization Summary:")
    print(f"   Reference band (Green/PAN): mean={ref_mean:.2f}, std={ref_std:.2f}")
    print(f"   Tolerance: ±{args.tolerance*100:.0f}% ({ref_std*(1-args.tolerance):.2f} - {ref_std*(1+args.tolerance):.2f})")

    passed_bands = [band for band, passed in validation_results.items() if passed]
    failed_bands = [band for band, passed in validation_results.items() if not passed]

    if passed_bands:
        print(f"   PASSED validation: {', '.join(passed_bands)}")
    if failed_bands:
        print(f"   FAILED validation: {', '.join(failed_bands)}")

    print(f"\nProcessed {len(target_bands)} bands successfully.")
    print(f"Validation: {len(passed_bands)}/{len(target_bands)} bands within {args.tolerance*100:.0f}% tolerance")

    # Create comparison plot
    if args.plot and len(band_statistics) > 1:
        print(f"\n4. Creating statistics comparison plot...")
        create_statistics_comparison_plot(band_statistics)

if __name__ == "__main__":
    main()