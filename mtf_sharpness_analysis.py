#!/usr/bin/env python3
"""
MTF (Modulation Transfer Function) Sharpness Analysis

This tool estimates MTF by measuring average edge sharpness across the image
using gradient operators (Sobel, Prewitt). Higher average gradient magnitude
indicates better spatial resolution and MTF performance.
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
# Using scipy and numpy instead of cv2 for gradient operations

def read_geotiff_10bit(file_path):
    """Read a 10-bit GeoTIFF file and return the image data."""
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

def calculate_mtf_sobel(image_data, exclude_zeros=True, verbose=False):
    """
    Calculate MTF proxy using Sobel gradient operator.

    Args:
        image_data: Input image
        exclude_zeros: Whether to exclude zero pixels from calculation
        verbose: Enable detailed output

    Returns:
        dict: MTF statistics including mean gradient magnitude
    """
    if verbose:
        print(f"    Calculating MTF using Sobel operator...")

    # Prepare image data
    if exclude_zeros:
        valid_mask = image_data > 0
        if np.sum(valid_mask) == 0:
            return {'mtf_score': 0, 'method': 'sobel', 'valid_pixels': 0}
    else:
        valid_mask = np.ones_like(image_data, dtype=bool)

    # Convert to float for gradient calculation
    img_float = image_data.astype(np.float32)

    # Apply Sobel filters for X and Y gradients using scipy
    sobel_x = ndimage.sobel(img_float, axis=1)  # X-direction (horizontal)
    sobel_y = ndimage.sobel(img_float, axis=0)  # Y-direction (vertical)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Apply mask if excluding zeros
    if exclude_zeros:
        gradient_valid = gradient_magnitude[valid_mask]
    else:
        gradient_valid = gradient_magnitude.flatten()

    # Calculate MTF score (mean gradient magnitude)
    mtf_score = np.mean(gradient_valid) if len(gradient_valid) > 0 else 0

    # Additional statistics
    stats = {
        'mtf_score': mtf_score,
        'method': 'sobel',
        'valid_pixels': len(gradient_valid),
        'gradient_std': np.std(gradient_valid) if len(gradient_valid) > 0 else 0,
        'gradient_max': np.max(gradient_valid) if len(gradient_valid) > 0 else 0,
        'gradient_percentiles': {
            '50': np.percentile(gradient_valid, 50) if len(gradient_valid) > 0 else 0,
            '75': np.percentile(gradient_valid, 75) if len(gradient_valid) > 0 else 0,
            '90': np.percentile(gradient_valid, 90) if len(gradient_valid) > 0 else 0,
            '95': np.percentile(gradient_valid, 95) if len(gradient_valid) > 0 else 0
        }
    }

    if verbose:
        print(f"    MTF score (mean gradient): {mtf_score:.3f}")
        print(f"    Gradient std: {stats['gradient_std']:.3f}")
        print(f"    Valid pixels: {stats['valid_pixels']}")

    return stats

def calculate_mtf_prewitt(image_data, exclude_zeros=True, verbose=False):
    """
    Calculate MTF proxy using Prewitt gradient operator.
    """
    if verbose:
        print(f"    Calculating MTF using Prewitt operator...")

    # Prepare image data
    if exclude_zeros:
        valid_mask = image_data > 0
        if np.sum(valid_mask) == 0:
            return {'mtf_score': 0, 'method': 'prewitt', 'valid_pixels': 0}
    else:
        valid_mask = np.ones_like(image_data, dtype=bool)

    # Convert to float
    img_float = image_data.astype(np.float32)

    # Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # Apply Prewitt filters using scipy
    grad_x = ndimage.convolve(img_float, prewitt_x)
    grad_y = ndimage.convolve(img_float, prewitt_y)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Apply mask if excluding zeros
    if exclude_zeros:
        gradient_valid = gradient_magnitude[valid_mask]
    else:
        gradient_valid = gradient_magnitude.flatten()

    # Calculate MTF score
    mtf_score = np.mean(gradient_valid) if len(gradient_valid) > 0 else 0

    stats = {
        'mtf_score': mtf_score,
        'method': 'prewitt',
        'valid_pixels': len(gradient_valid),
        'gradient_std': np.std(gradient_valid) if len(gradient_valid) > 0 else 0,
        'gradient_max': np.max(gradient_valid) if len(gradient_valid) > 0 else 0
    }

    if verbose:
        print(f"    MTF score (mean gradient): {mtf_score:.3f}")

    return stats

def calculate_mtf_laplacian(image_data, exclude_zeros=True, verbose=False):
    """
    Calculate MTF proxy using Laplacian operator (measures second derivatives).
    """
    if verbose:
        print(f"    Calculating MTF using Laplacian operator...")

    # Prepare image data
    if exclude_zeros:
        valid_mask = image_data > 0
        if np.sum(valid_mask) == 0:
            return {'mtf_score': 0, 'method': 'laplacian', 'valid_pixels': 0}
    else:
        valid_mask = np.ones_like(image_data, dtype=bool)

    # Convert to float
    img_float = image_data.astype(np.float32)

    # Apply Laplacian filter using scipy
    laplacian = ndimage.laplace(img_float)
    laplacian_abs = np.abs(laplacian)

    # Apply mask if excluding zeros
    if exclude_zeros:
        laplacian_valid = laplacian_abs[valid_mask]
    else:
        laplacian_valid = laplacian_abs.flatten()

    # Calculate MTF score
    mtf_score = np.mean(laplacian_valid) if len(laplacian_valid) > 0 else 0

    stats = {
        'mtf_score': mtf_score,
        'method': 'laplacian',
        'valid_pixels': len(laplacian_valid),
        'gradient_std': np.std(laplacian_valid) if len(laplacian_valid) > 0 else 0,
        'gradient_max': np.max(laplacian_valid) if len(laplacian_valid) > 0 else 0
    }

    if verbose:
        print(f"    MTF score (mean Laplacian): {mtf_score:.3f}")

    return stats

def calculate_comprehensive_mtf(image_data, band_name, methods=['sobel', 'prewitt', 'laplacian'], verbose=False):
    """
    Calculate MTF using multiple gradient operators for comprehensive analysis.
    """
    if verbose:
        print(f"\n  Analyzing {band_name} band sharpness/MTF...")

    mtf_results = {'band_name': band_name}

    # Calculate using different methods
    for method in methods:
        if method == 'sobel':
            result = calculate_mtf_sobel(image_data, verbose=verbose)
        elif method == 'prewitt':
            result = calculate_mtf_prewitt(image_data, verbose=verbose)
        elif method == 'laplacian':
            result = calculate_mtf_laplacian(image_data, verbose=verbose)
        else:
            continue

        mtf_results[method] = result

    # Calculate consensus MTF score (average of methods)
    valid_scores = [mtf_results[method]['mtf_score'] for method in methods if method in mtf_results]
    mtf_results['consensus_mtf'] = np.mean(valid_scores) if valid_scores else 0

    if verbose:
        print(f"    Consensus MTF score: {mtf_results['consensus_mtf']:.3f}")

    return mtf_results

def create_gradient_visualization(image_data, band_name, method='sobel', save_plot=True):
    """
    Create visualization showing original image and gradient map.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    ax1.imshow(image_data, cmap='gray', vmin=0, vmax=np.percentile(image_data[image_data > 0], 99))
    ax1.set_title(f'{band_name} Band - Original Image')
    ax1.axis('off')

    # Calculate and display gradient
    img_float = image_data.astype(np.float32)

    if method == 'sobel':
        sobel_x = ndimage.sobel(img_float, axis=1)
        sobel_y = ndimage.sobel(img_float, axis=0)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        title_suffix = "Sobel Gradient"
    elif method == 'prewitt':
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = ndimage.convolve(img_float, prewitt_x)
        grad_y = ndimage.convolve(img_float, prewitt_y)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        title_suffix = "Prewitt Gradient"
    else:  # laplacian
        gradient = np.abs(ndimage.laplace(img_float))
        title_suffix = "Laplacian"

    # Display gradient map
    im2 = ax2.imshow(gradient, cmap='hot', vmin=0, vmax=np.percentile(gradient, 95))
    ax2.set_title(f'{band_name} - {title_suffix} Map')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Gradient Magnitude')

    # Histogram of gradient values
    valid_mask = image_data > 0
    gradient_valid = gradient[valid_mask]

    ax3.hist(gradient_valid, bins=50, alpha=0.7, color='blue', density=True)
    ax3.axvline(np.mean(gradient_valid), color='red', linestyle='--', linewidth=2,
                label=f'Mean (MTF): {np.mean(gradient_valid):.3f}')
    ax3.set_xlabel('Gradient Magnitude')
    ax3.set_ylabel('Density')
    ax3.set_title(f'{band_name} - Gradient Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        output_file = f"mtf_gradient_{band_name.lower()}_{method}.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Gradient visualization saved: {output_file}")

    plt.show()

def create_mtf_comparison_plot(mtf_results, output_filename="mtf_sharpness_analysis.png"):
    """
    Create comprehensive MTF comparison visualization.
    """
    bands = list(mtf_results.keys())
    if not bands:
        return

    # Extract data for plotting
    consensus_scores = [mtf_results[band]['consensus_mtf'] for band in bands]
    sobel_scores = [mtf_results[band].get('sobel', {}).get('mtf_score', 0) for band in bands]
    prewitt_scores = [mtf_results[band].get('prewitt', {}).get('mtf_score', 0) for band in bands]
    laplacian_scores = [mtf_results[band].get('laplacian', {}).get('mtf_score', 0) for band in bands]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MTF (Sharpness) Analysis Results\nSpatial Resolution Quality Assessment',
                 fontsize=16, fontweight='bold')

    x = np.arange(len(bands))
    width = 0.25

    # Plot 1: Comparison of different gradient methods
    bars1 = ax1.bar(x - width, sobel_scores, width, label='Sobel', alpha=0.8, color='blue')
    bars2 = ax1.bar(x, prewitt_scores, width, label='Prewitt', alpha=0.8, color='green')
    bars3 = ax1.bar(x + width, laplacian_scores, width, label='Laplacian', alpha=0.8, color='red')

    ax1.set_xlabel('Spectral Bands')
    ax1.set_ylabel('MTF Score (Average Gradient)')
    ax1.set_title('MTF Scores by Gradient Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (sobel, prewitt, lap) in enumerate(zip(sobel_scores, prewitt_scores, laplacian_scores)):
        ax1.text(i - width, sobel + max(sobel_scores)*0.01, f'{sobel:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i, prewitt + max(prewitt_scores)*0.01, f'{prewitt:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width, lap + max(laplacian_scores)*0.01, f'{lap:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Consensus MTF scores
    bars4 = ax2.bar(x, consensus_scores, color='purple', alpha=0.7)

    ax2.set_xlabel('Spectral Bands')
    ax2.set_ylabel('Consensus MTF Score')
    ax2.set_title('Overall MTF Performance by Band')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels and rank
    sorted_indices = np.argsort(consensus_scores)[::-1]  # Descending order
    for i, score in enumerate(consensus_scores):
        rank = np.where(sorted_indices == i)[0][0] + 1
        ax2.text(i, score + max(consensus_scores)*0.01, f'{score:.3f}\n#{rank}',
                ha='center', va='bottom', fontweight='bold')

    # Plot 3: MTF score distribution
    all_scores = []
    all_labels = []
    colors = ['blue', 'green', 'red']
    methods = ['sobel', 'prewitt', 'laplacian']

    for j, method in enumerate(methods):
        method_scores = [mtf_results[band].get(method, {}).get('mtf_score', 0) for band in bands]
        all_scores.extend(method_scores)
        all_labels.extend([method] * len(method_scores))

    # Box plot of scores by method
    method_data = [
        [mtf_results[band].get(method, {}).get('mtf_score', 0) for band in bands]
        for method in methods
    ]

    ax3.boxplot(method_data, labels=methods, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax3.set_ylabel('MTF Score')
    ax3.set_title('MTF Score Distribution by Method')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Relative performance (normalized)
    # Normalize scores to show relative performance
    max_consensus = max(consensus_scores) if consensus_scores else 1
    normalized_scores = [score / max_consensus for score in consensus_scores]

    bars5 = ax4.bar(x, normalized_scores, color='orange', alpha=0.7)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Best performance')
    ax4.axhline(y=0.8, color='orange', linestyle=':', alpha=0.7, label='80% of best')

    ax4.set_xlabel('Spectral Bands')
    ax4.set_ylabel('Relative MTF Performance')
    ax4.set_title('Normalized MTF Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bands, rotation=45)
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add percentage labels
    for i, norm_score in enumerate(normalized_scores):
        ax4.text(i, norm_score + 0.02, f'{norm_score*100:.0f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"MTF analysis plot saved: {output_filename}")
    plt.show()

def main():
    """Main function for MTF sharpness analysis."""
    parser = argparse.ArgumentParser(description='MTF (sharpness) analysis using gradient operators')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_individual_enhanced_*.tiff')
    parser.add_argument('--methods', nargs='+', choices=['sobel', 'prewitt', 'laplacian'],
                       default=['sobel', 'prewitt', 'laplacian'], help='Gradient methods to use')
    parser.add_argument('--compare-normalized', action='store_true',
                       help='Compare with 16-bit normalized versions')
    parser.add_argument('--visualize-gradients', action='store_true',
                       help='Create gradient visualization plots')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Find input files
    current_dir = Path('.')
    band_files = {}
    expected_bands = ['green', 'nir', 'blue', 'red_edge', 'red']

    print(f"MTF (Modulation Transfer Function) Sharpness Analysis")
    print(f"Using spatially aligned individual enhanced images")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Using gradient operators to estimate spatial resolution")
    print()

    for band in expected_bands:
        pattern = args.input_pattern.replace('*', band)
        matching_files = list(current_dir.glob(pattern))
        if matching_files:
            band_files[band] = matching_files[0]
            print(f"Found {band}: {matching_files[0]}")

    if not band_files:
        print("No input files found!")
        return

    # Analyze spatially aligned individual enhanced images
    print(f"\n1. Analyzing spatially aligned individual enhanced images for MTF/sharpness...")
    original_mtf_results = {}

    for band_name, file_path in band_files.items():
        image_data = read_geotiff_10bit(file_path)
        mtf_stats = calculate_comprehensive_mtf(image_data, band_name, args.methods, args.verbose)
        original_mtf_results[band_name] = mtf_stats

        # Create gradient visualizations if requested
        if args.visualize_gradients:
            create_gradient_visualization(image_data, band_name, method='sobel')

    # Compare with normalized versions if requested
    normalized_mtf_results = {}
    if args.compare_normalized:
        print(f"\n2. Analyzing 16-bit normalized images for MTF/sharpness...")

        for band_name in band_files.keys():
            if band_name == 'green_pan':
                continue  # Skip reference band

            normalized_file = f"pushbroom_aligned_{band_name}_236images_start0_25pxsec_greenpan_16bit_normalized.tiff"
            normalized_path = current_dir / normalized_file

            if normalized_path.exists():
                # Read as 16-bit format, convert to 10-bit equivalent for comparison
                with Image.open(normalized_path) as img:
                    normalized_data = np.array(img) / 16.0

                mtf_stats = calculate_comprehensive_mtf(normalized_data, f"{band_name}_normalized",
                                                      args.methods, args.verbose)
                normalized_mtf_results[band_name] = mtf_stats
            else:
                print(f"    Normalized file not found: {normalized_file}")

    # Results summary
    print(f"\n3. MTF ANALYSIS SUMMARY:")
    print(f"   Methods used: {', '.join(args.methods)}")
    print(f"   Higher scores indicate better spatial resolution/sharpness")
    print()

    print("Spatially Aligned Individual Enhanced Images MTF Scores:")
    for band_name in expected_bands:
        if band_name in original_mtf_results:
            consensus = original_mtf_results[band_name]['consensus_mtf']
            sobel = original_mtf_results[band_name].get('sobel', {}).get('mtf_score', 0)
            print(f"  {band_name:12}: Consensus = {consensus:6.3f}, Sobel = {sobel:6.3f}")

    if normalized_mtf_results:
        print("\nAfter 16-bit Normalization:")
        for band_name in normalized_mtf_results.keys():
            if band_name in original_mtf_results:
                orig_consensus = original_mtf_results[band_name]['consensus_mtf']
                norm_consensus = normalized_mtf_results[band_name]['consensus_mtf']
                change = norm_consensus / orig_consensus if orig_consensus > 0 else 0
                print(f"  {band_name:12}: Consensus = {norm_consensus:6.3f} ({change:5.2f}x)")

    # Rank bands by sharpness
    print(f"\n4. SHARPNESS RANKING (Spatially Aligned Enhanced Images):")
    band_scores = [(band, original_mtf_results[band]['consensus_mtf'])
                  for band in original_mtf_results.keys()]
    band_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (band, score) in enumerate(band_scores, 1):
        print(f"   #{rank}: {band:12} (MTF: {score:.3f})")

    # Create visualizations
    print(f"\n5. Creating MTF analysis plots...")
    create_mtf_comparison_plot(original_mtf_results, "mtf_spatially_aligned_analysis.png")

    if normalized_mtf_results:
        create_mtf_comparison_plot(normalized_mtf_results, "mtf_normalized_analysis.png")

if __name__ == "__main__":
    main()