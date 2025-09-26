#!/usr/bin/env python3
"""
Statistical Normalization Using Full 16-bit Storage Range

This version leverages the fact that 10-bit data is stored in 16-bit format,
giving us 4x more dynamic range to achieve perfect standard deviation matching.
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def read_geotiff_as_stored(file_path):
    """
    Read GeoTIFF maintaining the 16-bit storage format.
    This preserves the full dynamic range available.
    """
    with Image.open(file_path) as img:
        img_array = np.array(img)
        return img_array  # Keep as 16-bit storage format

def calculate_statistics_exclude_zeros(image_data, verbose=False):
    """Calculate statistics excluding zeros from 16-bit storage format data."""
    valid_mask = image_data > 0
    valid_pixels = image_data[valid_mask]

    if len(valid_pixels) == 0:
        return 0.0, 0.0, 0

    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)

    if verbose:
        print(f"    Valid pixels: {len(valid_pixels)}/{image_data.size} ({len(valid_pixels)/image_data.size*100:.1f}%)")
        print(f"    16-bit storage stats: mean={mean_val:.1f}, std={std_val:.1f}")
        print(f"    10-bit equivalent: mean={mean_val/16:.2f}, std={std_val/16:.2f}")
        print(f"    Range: {valid_pixels.min()} - {valid_pixels.max()} (max possible: 65535)")

    return mean_val, std_val, len(valid_pixels)

def apply_normalization_16bit_range(image_data, target_mean, target_std, source_mean, source_std, verbose=False):
    """
    Apply normalization using the full 16-bit storage range.
    This should achieve perfect standard deviation matching without clipping.
    """
    if verbose:
        print(f"    16-bit range normalization:")
        print(f"    Target: mean={target_mean:.1f}, std={target_std:.1f}")
        print(f"    Source: mean={source_mean:.1f}, std={source_std:.1f}")

    if source_std == 0:
        print("    Warning: Source std is 0, returning original")
        return image_data.copy(), {}

    valid_mask = image_data > 0
    result = image_data.copy().astype(np.float32)

    # Apply perfect normalization formula in 16-bit space
    normalized_valid = ((image_data[valid_mask].astype(np.float32) - source_mean) *
                       (target_std / source_std)) + target_mean

    # Check if results fit in 16-bit range
    result_min, result_max = normalized_valid.min(), normalized_valid.max()
    fits_in_16bit = (result_min >= 0) and (result_max <= 65535)

    if fits_in_16bit:
        # Perfect! No clipping needed
        result[valid_mask] = normalized_valid
        clipping_info = {
            'clipped': False,
            'fits_in_16bit': True,
            'range': (result_min, result_max),
            'headroom_used': result_max / 65535
        }
        if verbose:
            print(f"    Perfect fit! Range: {result_min:.1f} - {result_max:.1f}")
            print(f"    16-bit headroom used: {result_max/65535*100:.1f}%")
    else:
        # Need to handle overflow - use smart clipping/scaling
        if result_max > 65535:
            # Scale down to fit while preserving relationships
            scale_factor = 65535 / result_max
            scaled_valid = (normalized_valid - target_mean) * scale_factor + target_mean
            result[valid_mask] = np.clip(scaled_valid, 0, 65535)

            clipping_info = {
                'clipped': True,
                'fits_in_16bit': False,
                'scale_factor': scale_factor,
                'original_range': (result_min, result_max),
                'final_range': (result[valid_mask].min(), result[valid_mask].max())
            }
            if verbose:
                print(f"    Scaled by {scale_factor:.3f} to fit 16-bit range")
        else:
            # Just clip negative values
            result[valid_mask] = np.clip(normalized_valid, 0, 65535)
            clipping_info = {
                'clipped': True,
                'fits_in_16bit': True,
                'negative_clipped': np.sum(normalized_valid < 0),
                'range': (result[valid_mask].min(), result[valid_mask].max())
            }

    # Convert back to uint16 for storage
    result = result.astype(np.uint16)

    # Calculate final statistics
    final_mean, final_std, _ = calculate_statistics_exclude_zeros(result)

    # Check if std target achieved
    std_target_achieved = abs(final_std - target_std) <= target_std * 0.1

    info = {
        **clipping_info,
        'final_mean': final_mean,
        'final_std': final_std,
        'std_target_achieved': std_target_achieved,
        'std_error': abs(final_std - target_std)
    }

    if verbose:
        print(f"    Final: mean={final_mean:.1f}, std={final_std:.1f}")
        print(f"    10-bit equiv: mean={final_mean/16:.2f}, std={final_std/16:.2f}")
        print(f"    Std target achieved: {'YES' if std_target_achieved else 'NO'}")

    return result, info

def save_16bit_tiff(img_array, output_path, verbose=False):
    """Save 16-bit image data maintaining the storage format."""
    Image.fromarray(img_array.astype(np.uint16)).save(output_path)
    if verbose:
        print(f"    Saved: {output_path}")
        print(f"    16-bit range: {img_array.min()} - {img_array.max()}")

def create_16bit_range_plot(band_statistics, target_std, tolerance=0.1):
    """Create visualization showing 16-bit range utilization and results."""
    bands = [b for b in band_statistics.keys() if b != 'green_pan']
    if not bands:
        return

    # Prepare data
    original_stds = [band_statistics[band]['original_std'] for band in bands]
    normalized_stds = [band_statistics[band]['normalized_std'] for band in bands]
    std_achieved = [band_statistics[band]['std_achieved'] for band in bands]
    headroom_used = [band_statistics[band].get('headroom_used', 0) for band in bands]

    # Convert to 10-bit equivalent for display
    original_stds_10bit = [s/16 for s in original_stds]
    normalized_stds_10bit = [s/16 for s in normalized_stds]
    target_std_10bit = target_std / 16

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('16-bit Range Statistical Normalization Results\nLeveraging Full Storage Dynamic Range',
                 fontsize=16, fontweight='bold')

    x = np.arange(len(bands))
    width = 0.4

    # Plot 1: Standard deviation in 10-bit equivalent scale
    colors_std = ['green' if achieved else 'red' for achieved in std_achieved]
    bars1 = ax1.bar(x - width/2, original_stds_10bit, width, label='Original', alpha=0.8, color='lightblue')
    bars2 = ax1.bar(x + width/2, normalized_stds_10bit, width, label='Normalized', alpha=0.8, color=colors_std)

    # Add target line and tolerance
    ax1.axhline(y=target_std_10bit, color='red', linestyle='-', linewidth=2, label=f'Target ({target_std_10bit:.1f})')
    ax1.axhspan(target_std_10bit * (1-tolerance), target_std_10bit * (1+tolerance),
               alpha=0.2, color='red', label=f'±{tolerance*100:.0f}% tolerance')

    ax1.set_xlabel('Spectral Bands')
    ax1.set_ylabel('Standard Deviation (10-bit equivalent)')
    ax1.set_title('Standard Deviation Achievement (10-bit scale)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (orig, norm, achieved) in enumerate(zip(original_stds_10bit, normalized_stds_10bit, std_achieved)):
        ax1.text(i - width/2, orig + max(original_stds_10bit)*0.01, f'{orig:.0f}', ha='center', va='bottom')
        symbol = '✓' if achieved else '✗'
        color = 'green' if achieved else 'red'
        ax1.text(i + width/2, norm + max(normalized_stds_10bit)*0.01, f'{norm:.0f}{symbol}',
                ha='center', va='bottom', color=color, fontweight='bold')

    # Plot 2: 16-bit range utilization
    bars3 = ax2.bar(x, [h*100 for h in headroom_used], color='purple', alpha=0.7)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='16-bit limit')

    ax2.set_xlabel('Spectral Bands')
    ax2.set_ylabel('16-bit Range Utilization (%)')
    ax2.set_title('Dynamic Range Usage in 16-bit Storage')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add percentage labels
    for i, usage in enumerate(headroom_used):
        ax2.text(i, usage*100 + 2, f'{usage*100:.1f}%', ha='center', va='bottom')

    # Plot 3: Success validation
    colors_val = ['green' if achieved else 'red' for achieved in std_achieved]
    bars4 = ax3.bar(x, [1 if achieved else 0 for achieved in std_achieved], color=colors_val, alpha=0.7)

    ax3.set_xlabel('Spectral Bands')
    ax3.set_ylabel('10% Tolerance Achievement')
    ax3.set_title('Standard Deviation Target Achievement')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bands, rotation=45)
    ax3.set_ylim(0, 1.2)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['FAIL', 'PASS'])
    ax3.grid(True, alpha=0.3)

    # Add std values
    for i, (std_10bit, achieved) in enumerate(zip(normalized_stds_10bit, std_achieved)):
        status = 'PASS' if achieved else 'FAIL'
        ax3.text(i, 0.5, f'{std_10bit:.1f}\n{status}', ha='center', va='center',
                fontweight='bold', color='white')

    # Plot 4: Overall summary
    success_count = sum(std_achieved)
    total_count = len(std_achieved)

    summary_data = [success_count, total_count - success_count]
    summary_labels = [f'Success ({success_count})', f'Fail ({total_count - success_count})']
    summary_colors = ['green', 'red']

    ax4.pie(summary_data, labels=summary_labels, colors=summary_colors, autopct='%1.0f%%', startangle=90)
    ax4.set_title(f'16-bit Range Normalization\nOverall Success Rate')

    plt.tight_layout()
    output_file = "16bit_range_normalization_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"16-bit range normalization plot saved: {output_file}")
    plt.show()

def main():
    """Main function using full 16-bit storage range for normalization."""
    parser = argparse.ArgumentParser(description='Statistical normalization using full 16-bit storage range')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_aligned_*_236images_start0_25pxsec_greenpan.tiff')
    parser.add_argument('--output-suffix', type=str, default='_16bit_normalized')
    parser.add_argument('--tolerance', type=float, default=0.1)
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Find input files
    current_dir = Path('.')
    band_files = {}
    expected_bands = ['green_pan', 'nir', 'blue', 'red_edge', 'red']

    print(f"16-bit Range Statistical Normalization")
    print(f"Leveraging full 16-bit storage range (up to 65535)")
    print(f"Target: ±{args.tolerance*100:.0f}% standard deviation match")

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

    # Calculate reference statistics in 16-bit space
    print("\n1. Analyzing reference band (Green/PAN) in 16-bit storage format...")
    ref_image = read_geotiff_as_stored(band_files['green_pan'])
    ref_mean, ref_std, ref_count = calculate_statistics_exclude_zeros(ref_image, args.verbose)

    print(f"   Green/PAN 16-bit: mean={ref_mean:.1f}, std={ref_std:.1f}")
    print(f"   10-bit equivalent: mean={ref_mean/16:.2f}, std={ref_std/16:.2f}")

    # Target range
    std_target_min = ref_std * (1 - args.tolerance)
    std_target_max = ref_std * (1 + args.tolerance)
    print(f"   Target std range (16-bit): {std_target_min:.1f} - {std_target_max:.1f}")

    # Store results
    band_statistics = {
        'green_pan': {
            'original_std': ref_std, 'normalized_std': ref_std,
            'std_achieved': True
        }
    }

    # Process each band using 16-bit range
    print("\n2. Processing bands with 16-bit range normalization...")
    success_count = 0

    for band_name in target_bands:
        print(f"\n  Processing {band_name}...")

        # Read in 16-bit storage format
        target_image = read_geotiff_as_stored(band_files[band_name])
        orig_mean, orig_std, orig_count = calculate_statistics_exclude_zeros(target_image, args.verbose)

        # Apply normalization using full 16-bit range
        normalized_image, norm_info = apply_normalization_16bit_range(
            target_image, ref_mean, ref_std, orig_mean, orig_std, args.verbose)

        # Check results
        std_achieved = norm_info['std_target_achieved']

        if std_achieved:
            success_count += 1
            print(f"    ✅ SUCCESS: Achieved 10% standard deviation target!")
        else:
            print(f"    ❌ MISS: Std error = {norm_info['std_error']:.1f}")

        # Store statistics
        band_statistics[band_name] = {
            'original_std': orig_std,
            'normalized_std': norm_info['final_std'],
            'std_achieved': std_achieved,
            'headroom_used': norm_info.get('headroom_used', norm_info['final_std']/65535)
        }

        # Save result
        input_path = band_files[band_name]
        output_path = input_path.stem + args.output_suffix + input_path.suffix
        save_16bit_tiff(normalized_image, output_path, args.verbose)

    # Final summary
    print(f"\n3. 16-BIT RANGE NORMALIZATION SUMMARY:")
    print(f"   🎯 SUCCESS: {success_count}/{len(target_bands)} bands achieved ±{args.tolerance*100:.0f}% std target")
    print(f"   📊 Reference std: {ref_std:.1f} (16-bit) = {ref_std/16:.2f} (10-bit equiv)")
    print(f"   🚀 Dynamic range: Using up to 65535 (4x standard 10-bit range)")

    if success_count == len(target_bands):
        print("   🏆 PERFECT: All bands achieved exact 10% tolerance using 16-bit range!")
    elif success_count > 0:
        print(f"   📈 PARTIAL: {success_count} bands achieved target")

    # Detailed analysis
    print(f"\n4. Range utilization analysis:")
    for band_name in target_bands:
        if band_name in band_statistics:
            usage = band_statistics[band_name]['headroom_used']
            print(f"   {band_name}: {usage*100:.1f}% of 16-bit range used")

    # Create visualization
    print(f"\n5. Creating 16-bit range analysis plot...")
    create_16bit_range_plot(band_statistics, ref_std, args.tolerance)

if __name__ == "__main__":
    main()