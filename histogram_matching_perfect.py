#!/usr/bin/env python3
"""
Perfect Statistical Normalization with Extended Range Options

This version achieves perfect 10% standard deviation matching by providing
multiple solutions for handling the range constraint issue.
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

def calculate_statistics_exclude_zeros(image_data, verbose=False):
    """Calculate mean and standard deviation excluding zeros."""
    valid_mask = image_data > 0
    valid_pixels = image_data[valid_mask]

    if len(valid_pixels) == 0:
        return 0.0, 0.0, 0

    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)

    if verbose:
        print(f"    Valid pixels: {len(valid_pixels)}/{image_data.size} ({len(valid_pixels)/image_data.size*100:.1f}%)")
        print(f"    Statistics: mean={mean_val:.2f}, std={std_val:.2f}")

    return mean_val, std_val, len(valid_pixels)

def perfect_normalization_with_range_options(image_data, target_mean, target_std, source_mean, source_std,
                                            range_strategy='extended', verbose=False):
    """
    Apply perfect normalization with different strategies for handling range constraints.

    Range strategies:
    - 'extended': Save as extended range (>10-bit), perfect std matching
    - 'rescaled': Rescale the entire dataset to fit 10-bit while preserving ratios
    - 'compressed': Compress dynamic range to fit, partial std matching
    - 'hybrid': Optimize for best compromise between std matching and range compliance
    """
    if verbose:
        print(f"    Range strategy: {range_strategy}")
        print(f"    Target: mean={target_mean:.2f}, std={target_std:.2f}")
        print(f"    Source: mean={source_mean:.2f}, std={source_std:.2f}")

    if source_std == 0:
        return image_data.copy(), {'strategy': range_strategy, 'perfect_std_match': False}

    valid_mask = image_data > 0
    result = image_data.copy().astype(np.float32)

    # Apply perfect normalization formula
    normalized_valid = ((image_data[valid_mask].astype(np.float32) - source_mean) *
                       (target_std / source_std)) + target_mean

    if range_strategy == 'extended':
        # Perfect normalization, may exceed 10-bit range
        result[valid_mask] = normalized_valid
        # Don't clip, save as extended range
        info = {
            'strategy': 'extended',
            'perfect_std_match': True,
            'range_min': normalized_valid.min(),
            'range_max': normalized_valid.max(),
            'exceeds_10bit': normalized_valid.max() > 1023 or normalized_valid.min() < 0
        }

    elif range_strategy == 'rescaled':
        # Rescale entire dataset to fit 10-bit range while preserving relationships
        data_min, data_max = normalized_valid.min(), normalized_valid.max()

        if data_max <= 1023 and data_min >= 0:
            # Already fits, no rescaling needed
            result[valid_mask] = normalized_valid
            rescale_factor = 1.0
        else:
            # Rescale to fit [0, 1023] while preserving center and relationships
            data_center = (data_max + data_min) / 2
            data_range = data_max - data_min
            target_range = 1023

            if data_range > 0:
                rescale_factor = target_range / data_range
                rescaled = (normalized_valid - data_center) * rescale_factor + 512  # Center at 512
                result[valid_mask] = np.clip(rescaled, 0, 1023)
            else:
                result[valid_mask] = normalized_valid
                rescale_factor = 1.0

        info = {
            'strategy': 'rescaled',
            'perfect_std_match': False,
            'rescale_factor': rescale_factor,
            'original_range': (data_min, data_max),
            'final_range': (result[valid_mask].min(), result[valid_mask].max())
        }

    elif range_strategy == 'compressed':
        # Compress dynamic range using percentile-based approach
        p5, p95 = np.percentile(normalized_valid, [5, 95])
        compressed_range = p95 - p5

        # Map the central 90% to most of the 10-bit range
        target_range = 900  # Use 900 out of 1023 to leave margins
        scale_factor = target_range / compressed_range if compressed_range > 0 else 1.0

        compressed = ((normalized_valid - p5) * scale_factor) + 50  # Start at 50
        result[valid_mask] = np.clip(compressed, 0, 1023)

        info = {
            'strategy': 'compressed',
            'perfect_std_match': False,
            'compression_percentiles': (p5, p95),
            'scale_factor': scale_factor
        }

    elif range_strategy == 'hybrid':
        # Optimize for best compromise between std matching and range compliance
        # Use 90% of the transformation to preserve most of the std improvement

        original_valid = image_data[valid_mask].astype(np.float32)
        hybrid_strength = 0.9  # 90% of full transformation

        # Blend original and normalized
        hybrid_valid = (original_valid * (1 - hybrid_strength) +
                       normalized_valid * hybrid_strength)

        # Ensure it fits in range
        result[valid_mask] = np.clip(hybrid_valid, 0, 1023)

        info = {
            'strategy': 'hybrid',
            'perfect_std_match': False,
            'hybrid_strength': hybrid_strength,
            'transformation_percent': hybrid_strength * 100
        }

    # Calculate final statistics
    final_mean, final_std, _ = calculate_statistics_exclude_zeros(result)
    info['final_mean'] = final_mean
    info['final_std'] = final_std
    info['std_achievement'] = abs(final_std - target_std) <= target_std * 0.1

    if verbose:
        print(f"    Result: mean={final_mean:.2f}, std={final_std:.2f}")
        print(f"    Std target achieved: {info['std_achievement']}")
        if 'exceeds_10bit' in info:
            print(f"    Exceeds 10-bit range: {info['exceeds_10bit']}")

    return result.astype(np.uint16) if range_strategy != 'extended' else result, info

def save_image_with_strategy(img_array, output_path, strategy_info, verbose=False):
    """Save image with appropriate format based on strategy."""
    if strategy_info['strategy'] == 'extended':
        # Save as 32-bit TIFF to preserve extended range
        from PIL import Image
        # Convert to 32-bit float for extended range
        img_32bit = img_array.astype(np.float32)
        # Save with PIL (will use 32-bit float TIFF)
        Image.fromarray(img_32bit, mode='F').save(output_path.replace('.tiff', '_extended.tiff'))

        if verbose:
            print(f"    Saved as 32-bit extended range: {output_path.replace('.tiff', '_extended.tiff')}")
            print(f"    Range: {img_array.min():.1f} to {img_array.max():.1f}")
    else:
        # Save as standard 16-bit (scaled back to storage format)
        img_16bit = (img_array * 16).astype(np.uint16)
        Image.fromarray(img_16bit).save(output_path)

        if verbose:
            print(f"    Saved as 16-bit standard: {output_path}")
            print(f"    Range: {img_array.min():.1f} to {img_array.max():.1f}")

def create_perfect_normalization_plot(band_statistics, target_std, tolerance=0.1):
    """Create comprehensive visualization of perfect normalization results."""
    bands = [b for b in band_statistics.keys() if b != 'green_pan']
    if not bands:
        return

    # Prepare data
    strategies = [band_statistics[band]['strategy'] for band in bands]
    original_stds = [band_statistics[band]['original_std'] for band in bands]
    normalized_stds = [band_statistics[band]['normalized_std'] for band in bands]
    std_achieved = [band_statistics[band]['std_achieved'] for band in bands]

    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Perfect Statistical Normalization Results\nAchieving 10% Standard Deviation Target',
                 fontsize=16, fontweight='bold')

    x = np.arange(len(bands))
    width = 0.4

    # Plot 1: Standard deviation achievement
    colors_std = ['green' if achieved else 'orange' for achieved in std_achieved]
    bars1 = ax1.bar(x - width/2, original_stds, width, label='Original', alpha=0.8, color='lightblue')
    bars2 = ax1.bar(x + width/2, normalized_stds, width, label='Normalized', alpha=0.8, color=colors_std)

    # Add target line and tolerance
    ax1.axhline(y=target_std, color='red', linestyle='-', linewidth=2, label=f'Target ({target_std:.0f})')
    ax1.axhspan(target_std * (1-tolerance), target_std * (1+tolerance),
               alpha=0.2, color='red', label=f'±{tolerance*100:.0f}% tolerance')

    ax1.set_xlabel('Spectral Bands')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Standard Deviation: Perfect Normalization Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (orig, norm, achieved) in enumerate(zip(original_stds, normalized_stds, std_achieved)):
        ax1.text(i - width/2, orig + max(original_stds)*0.01, f'{orig:.0f}', ha='center', va='bottom')
        color = 'green' if achieved else 'orange'
        symbol = '✓' if achieved else '~'
        ax1.text(i + width/2, norm + max(normalized_stds)*0.01, f'{norm:.0f}{symbol}',
                ha='center', va='bottom', color=color, fontweight='bold')

    # Plot 2: Strategy used
    strategy_colors = {
        'extended': 'purple', 'rescaled': 'blue',
        'compressed': 'orange', 'hybrid': 'green'
    }
    colors_strategy = [strategy_colors.get(s, 'gray') for s in strategies]
    bars3 = ax2.bar(x, [1]*len(bands), color=colors_strategy, alpha=0.7)

    ax2.set_xlabel('Spectral Bands')
    ax2.set_ylabel('Strategy Used')
    ax2.set_title('Normalization Strategy by Band')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])

    # Add strategy labels
    for i, strategy in enumerate(strategies):
        ax2.text(i, 0.5, strategy.title(), ha='center', va='center',
                fontweight='bold', color='white', rotation=45)

    # Add legend for strategies
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=strategy.title())
                      for strategy, color in strategy_colors.items()]
    ax2.legend(handles=legend_elements, loc='upper right')

    # Plot 3: Success rate by strategy
    strategy_success = {}
    for i, (strategy, achieved) in enumerate(zip(strategies, std_achieved)):
        if strategy not in strategy_success:
            strategy_success[strategy] = {'total': 0, 'success': 0}
        strategy_success[strategy]['total'] += 1
        if achieved:
            strategy_success[strategy]['success'] += 1

    strategy_names = list(strategy_success.keys())
    success_rates = [strategy_success[s]['success'] / strategy_success[s]['total']
                    for s in strategy_names]

    bars4 = ax3.bar(strategy_names, success_rates,
                   color=[strategy_colors.get(s, 'gray') for s in strategy_names], alpha=0.7)

    ax3.set_xlabel('Normalization Strategy')
    ax3.set_ylabel('Success Rate (10% std achieved)')
    ax3.set_title('Strategy Success Rate')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)

    # Add percentage labels
    for i, (name, rate) in enumerate(zip(strategy_names, success_rates)):
        ax3.text(i, rate + 0.02, f'{rate*100:.0f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Overall summary
    total_bands = len(bands)
    successful_bands = sum(std_achieved)
    summary_data = [successful_bands, total_bands - successful_bands]
    summary_labels = ['Success', 'Partial']
    summary_colors = ['green', 'orange']

    ax4.pie(summary_data, labels=summary_labels, colors=summary_colors, autopct='%1.0f%%',
           startangle=90)
    ax4.set_title(f'Overall Success Rate\n{successful_bands}/{total_bands} bands achieved ±10% std')

    plt.tight_layout()
    output_file = "perfect_normalization_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Perfect normalization results plot saved: {output_file}")
    plt.show()

def main():
    """Main function for perfect standard deviation normalization."""
    parser = argparse.ArgumentParser(description='Perfect normalization achieving 10% std deviation target')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_aligned_*_236images_start0_25pxsec_greenpan.tiff')
    parser.add_argument('--output-suffix', type=str, default='_perfect_normalized')
    parser.add_argument('--strategy', choices=['extended', 'rescaled', 'compressed', 'hybrid', 'auto'],
                       default='auto', help='Range handling strategy')
    parser.add_argument('--tolerance', type=float, default=0.1)
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Find input files
    current_dir = Path('.')
    band_files = {}
    expected_bands = ['green_pan', 'nir', 'blue', 'red_edge', 'red']

    print(f"Perfect Statistical Normalization")
    print(f"Target: ±{args.tolerance*100:.0f}% standard deviation match")
    print(f"Strategy: {args.strategy}")

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

    # Calculate reference statistics
    print("\n1. Calculating reference statistics...")
    ref_image = read_geotiff_10bit(band_files['green_pan'])
    ref_mean, ref_std, ref_count = calculate_statistics_exclude_zeros(ref_image, args.verbose)
    print(f"   Green/PAN: mean={ref_mean:.2f}, std={ref_std:.2f}")

    # Target range
    std_target_min = ref_std * (1 - args.tolerance)
    std_target_max = ref_std * (1 + args.tolerance)
    print(f"   Target std range: {std_target_min:.2f} - {std_target_max:.2f}")

    # Store results
    band_statistics = {
        'green_pan': {
            'original_std': ref_std, 'normalized_std': ref_std,
            'std_achieved': True, 'strategy': 'reference'
        }
    }

    # Process each band
    print("\n2. Applying perfect normalization...")
    success_count = 0

    for band_name in target_bands:
        print(f"\n  Processing {band_name}...")

        # Read and analyze
        target_image = read_geotiff_10bit(band_files[band_name])
        orig_mean, orig_std, orig_count = calculate_statistics_exclude_zeros(target_image, args.verbose)

        # Select strategy
        if args.strategy == 'auto':
            # Auto-select based on how much expansion is needed
            expansion_factor = ref_std / orig_std if orig_std > 0 else 1.0
            if expansion_factor > 8:  # Very high expansion needed
                strategy = 'extended'  # Only extended range can handle this
            elif expansion_factor > 3:  # High expansion
                strategy = 'rescaled'  # Try to fit by rescaling
            elif expansion_factor > 1.5:  # Moderate expansion
                strategy = 'hybrid'    # Compromise approach
            else:  # Low expansion
                strategy = 'rescaled'  # Should fit easily

            print(f"    Auto-selected strategy: {strategy} (expansion factor: {expansion_factor:.1f}x)")
        else:
            strategy = args.strategy

        # Apply perfect normalization
        normalized_image, strategy_info = perfect_normalization_with_range_options(
            target_image, ref_mean, ref_std, orig_mean, orig_std, strategy, args.verbose)

        # Check results
        final_std = strategy_info['final_std']
        std_achieved = strategy_info['std_achievement']

        print(f"    Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
        print(f"    Result: mean={strategy_info['final_mean']:.2f}, std={final_std:.2f}")
        print(f"    10% target achieved: {'YES ✓' if std_achieved else 'NO ✗'}")

        if std_achieved:
            success_count += 1

        # Store statistics
        band_statistics[band_name] = {
            'original_std': orig_std,
            'normalized_std': final_std,
            'std_achieved': std_achieved,
            'strategy': strategy
        }

        # Save result
        input_path = band_files[band_name]
        output_path = input_path.stem + args.output_suffix + input_path.suffix
        save_image_with_strategy(normalized_image, output_path, strategy_info, args.verbose)

    # Final summary
    print(f"\n3. PERFECT NORMALIZATION SUMMARY:")
    print(f"   🎯 SUCCESS: {success_count}/{len(target_bands)} bands achieved ±{args.tolerance*100:.0f}% std target")
    print(f"   📊 Reference std: {ref_std:.2f}")

    if success_count == len(target_bands):
        print("   🏆 PERFECT: All bands successfully normalized to 10% tolerance!")
    elif success_count > 0:
        print(f"   📈 PARTIAL: {success_count} bands achieved perfect target")

    # Create visualization
    print("\n4. Creating perfect normalization visualization...")
    create_perfect_normalization_plot(band_statistics, ref_std, args.tolerance)

if __name__ == "__main__":
    main()