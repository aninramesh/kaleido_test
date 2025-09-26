#!/usr/bin/env python3
"""
Signal-to-Noise Ratio Analysis from Homogeneous Regions

This tool properly calculates SNR by identifying homogeneous regions where
variation represents noise rather than scene content. This approach separates
true noise characteristics from legitimate scene variation.
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Simple K-means implementation to avoid sklearn dependency
def simple_kmeans(data, k, max_iters=100):
    """Simple K-means clustering without sklearn dependency."""
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        # Assign points to closest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels

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

def detect_homogeneous_regions_sliding_window(image_data, window_size=32, stride=16,
                                            homogeneity_threshold=0.05, min_regions=10, verbose=False):
    """
    Detect homogeneous regions using sliding window approach.

    Args:
        image_data: Input image
        window_size: Size of analysis window (pixels)
        stride: Step size for sliding window
        homogeneity_threshold: Maximum coefficient of variation for homogeneity
        min_regions: Minimum number of regions to find
        verbose: Enable detailed output
    """
    if verbose:
        print(f"    Detecting homogeneous regions...")
        print(f"    Window: {window_size}x{window_size}, stride: {stride}")
        print(f"    Homogeneity threshold: {homogeneity_threshold:.3f} (CV)")

    valid_mask = image_data > 0
    height, width = image_data.shape

    homogeneous_regions = []
    region_stats = []

    # Slide window across image
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # Extract window
            window = image_data[y:y+window_size, x:x+window_size]
            window_mask = valid_mask[y:y+window_size, x:x+window_size]

            # Skip if not enough valid pixels
            valid_pixels = window[window_mask]
            if len(valid_pixels) < window_size * window_size * 0.8:  # At least 80% valid
                continue

            # Calculate homogeneity metrics
            window_mean = np.mean(valid_pixels)
            window_std = np.std(valid_pixels)

            if window_mean > 0:
                cv = window_std / window_mean  # Coefficient of variation

                # Check if region is homogeneous
                if cv <= homogeneity_threshold:
                    homogeneous_regions.append({
                        'bbox': (x, y, x+window_size, y+window_size),
                        'mean': window_mean,
                        'std': window_std,
                        'cv': cv,
                        'pixel_count': len(valid_pixels),
                        'snr': window_mean / window_std if window_std > 0 else float('inf')
                    })
                    region_stats.append((window_mean, window_std, cv))

    if verbose:
        print(f"    Found {len(homogeneous_regions)} homogeneous regions")
        if len(homogeneous_regions) > 0:
            cvs = [r['cv'] for r in homogeneous_regions]
            print(f"    CV range: {min(cvs):.4f} - {max(cvs):.4f}")

    # If we don't have enough regions, relax threshold
    if len(homogeneous_regions) < min_regions and homogeneity_threshold < 0.2:
        if verbose:
            print(f"    Insufficient regions ({len(homogeneous_regions)}), relaxing threshold...")
        return detect_homogeneous_regions_sliding_window(
            image_data, window_size, stride, homogeneity_threshold * 1.5, min_regions, verbose)

    return homogeneous_regions

def detect_homogeneous_regions_clustering(image_data, n_clusters=8, window_size=32,
                                        homogeneity_threshold=0.05, verbose=False):
    """
    Detect homogeneous regions using K-means clustering to identify similar areas.
    """
    if verbose:
        print(f"    Clustering-based homogeneous region detection...")
        print(f"    Clusters: {n_clusters}, window: {window_size}x{window_size}")

    valid_mask = image_data > 0
    height, width = image_data.shape

    # Sample patches for clustering
    patches = []
    patch_locations = []

    for y in range(0, height - window_size + 1, window_size//2):
        for x in range(0, width - window_size + 1, window_size//2):
            window = image_data[y:y+window_size, x:x+window_size]
            window_mask = valid_mask[y:y+window_size, x:x+window_size]

            valid_pixels = window[window_mask]
            if len(valid_pixels) > window_size * window_size * 0.7:
                # Use statistical features for clustering
                features = [
                    np.mean(valid_pixels),
                    np.std(valid_pixels),
                    np.percentile(valid_pixels, 25),
                    np.percentile(valid_pixels, 75)
                ]
                patches.append(features)
                patch_locations.append((x, y, np.mean(valid_pixels), np.std(valid_pixels)))

    if len(patches) < n_clusters:
        if verbose:
            print(f"    Too few patches ({len(patches)}), falling back to sliding window")
        return detect_homogeneous_regions_sliding_window(image_data, window_size,
                                                        window_size//2, homogeneity_threshold, verbose=verbose)

    # Perform clustering
    patches_array = np.array(patches)
    cluster_labels = simple_kmeans(patches_array, n_clusters)

    # Find most homogeneous clusters
    homogeneous_regions = []

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if np.sum(cluster_mask) < 3:  # Need at least 3 patches
            continue

        cluster_patches = [patch_locations[i] for i in range(len(patch_locations)) if cluster_mask[i]]

        # Check intra-cluster homogeneity
        cluster_means = [p[2] for p in cluster_patches]
        cluster_stds = [p[3] for p in cluster_patches]

        mean_variation = np.std(cluster_means) / np.mean(cluster_means) if np.mean(cluster_means) > 0 else 1.0

        if mean_variation <= homogeneity_threshold:
            # This cluster represents homogeneous regions
            for x, y, patch_mean, patch_std in cluster_patches:
                if patch_std > 0:
                    cv = patch_std / patch_mean
                    if cv <= homogeneity_threshold * 2:  # Allow slightly higher CV within clusters
                        homogeneous_regions.append({
                            'bbox': (x, y, x+window_size, y+window_size),
                            'mean': patch_mean,
                            'std': patch_std,
                            'cv': cv,
                            'cluster_id': cluster_id,
                            'snr': patch_mean / patch_std,
                            'pixel_count': window_size * window_size
                        })

    if verbose:
        print(f"    Found {len(homogeneous_regions)} homogeneous regions from clustering")

    return homogeneous_regions

def calculate_snr_from_homogeneous_regions(homogeneous_regions, method='median', verbose=False):
    """
    Calculate SNR statistics from identified homogeneous regions.

    Args:
        homogeneous_regions: List of homogeneous region data
        method: 'median', 'mean', 'weighted_mean', or 'all'
    """
    if not homogeneous_regions:
        return {'snr': 0, 'count': 0, 'method': method}

    snrs = [region['snr'] for region in homogeneous_regions if region['snr'] != float('inf')]
    means = [region['mean'] for region in homogeneous_regions]
    stds = [region['std'] for region in homogeneous_regions]
    pixel_counts = [region['pixel_count'] for region in homogeneous_regions]

    if not snrs:
        return {'snr': 0, 'count': 0, 'method': method}

    snr_stats = {
        'count': len(snrs),
        'method': method,
        'regions_used': len(homogeneous_regions),
        'individual_snrs': snrs,
        'region_means': means,
        'region_stds': stds
    }

    if method == 'median':
        snr_stats['snr'] = np.median(snrs)
    elif method == 'mean':
        snr_stats['snr'] = np.mean(snrs)
    elif method == 'weighted_mean':
        # Weight by pixel count
        weights = np.array(pixel_counts)
        snr_stats['snr'] = np.average(snrs, weights=weights)
    elif method == 'pooled':
        # Pool all pixels from homogeneous regions
        total_mean = np.average(means, weights=pixel_counts)
        pooled_variance = np.average(np.array(stds)**2, weights=pixel_counts)
        pooled_std = np.sqrt(pooled_variance)
        snr_stats['snr'] = total_mean / pooled_std if pooled_std > 0 else 0
        snr_stats['pooled_mean'] = total_mean
        snr_stats['pooled_std'] = pooled_std

    if verbose:
        print(f"    SNR calculation ({method}): {snr_stats['snr']:.2f}")
        print(f"    Based on {snr_stats['count']} homogeneous regions")
        print(f"    SNR range: {min(snrs):.2f} - {max(snrs):.2f}")

    return snr_stats

def analyze_band_snr(image_data, band_name, detection_method='sliding_window', snr_method='pooled',
                     window_size=32, stride=16, homogeneity_threshold=0.1, verbose=False):
    """
    Complete SNR analysis for a spectral band.
    """
    if verbose:
        print(f"\n  Analyzing {band_name} band SNR...")
        print(f"  Detection: {detection_method}, SNR calculation: {snr_method}")

    # Detect homogeneous regions
    if detection_method == 'sliding_window':
        regions = detect_homogeneous_regions_sliding_window(image_data, window_size=window_size,
                                                          stride=stride,
                                                          homogeneity_threshold=homogeneity_threshold,
                                                          verbose=verbose)
    elif detection_method == 'clustering':
        regions = detect_homogeneous_regions_clustering(image_data, window_size=window_size,
                                                       homogeneity_threshold=homogeneity_threshold,
                                                       verbose=verbose)
    else:
        raise ValueError(f"Unknown detection method: {detection_method}")

    # Calculate SNR from homogeneous regions
    snr_stats = calculate_snr_from_homogeneous_regions(regions, method=snr_method, verbose=verbose)

    # Add overall image statistics for comparison
    valid_mask = image_data > 0
    valid_pixels = image_data[valid_mask]

    if len(valid_pixels) > 0:
        overall_mean = np.mean(valid_pixels)
        overall_std = np.std(valid_pixels)
        overall_snr = overall_mean / overall_std if overall_std > 0 else 0
    else:
        overall_mean = overall_std = overall_snr = 0

    snr_stats.update({
        'band_name': band_name,
        'overall_snr': overall_snr,  # This includes scene variation (incorrect for SNR)
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'valid_pixels': len(valid_pixels),
        'homogeneous_regions': regions
    })

    if verbose:
        print(f"    True SNR (homogeneous regions): {snr_stats['snr']:.2f}")
        print(f"    Overall scene variation: {overall_snr:.2f} (includes scene content)")
        print(f"    SNR improvement over scene variation: {snr_stats['snr']/overall_snr:.2f}x")

    return snr_stats

def create_snr_analysis_plot(snr_results, output_filename="snr_homogeneous_analysis.png"):
    """
    Create comprehensive SNR analysis visualization.
    """
    bands = list(snr_results.keys())
    if not bands:
        return

    # Prepare data
    true_snrs = [snr_results[band]['snr'] for band in bands]
    overall_snrs = [snr_results[band]['overall_snr'] for band in bands]
    region_counts = [snr_results[band]['count'] for band in bands]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Signal-to-Noise Ratio Analysis from Homogeneous Regions\nSeparating True Noise from Scene Variation',
                 fontsize=16, fontweight='bold')

    x = np.arange(len(bands))
    width = 0.4

    # Plot 1: True SNR vs Scene Variation
    bars1 = ax1.bar(x - width/2, true_snrs, width, label='True SNR (homogeneous)', alpha=0.8, color='green')
    bars2 = ax1.bar(x + width/2, overall_snrs, width, label='Scene variation (misleading)', alpha=0.8, color='red')

    ax1.set_xlabel('Spectral Bands')
    ax1.set_ylabel('Signal-to-Noise Ratio')
    ax1.set_title('True SNR vs Scene Variation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (true_snr, scene_snr) in enumerate(zip(true_snrs, overall_snrs)):
        ax1.text(i - width/2, true_snr + max(true_snrs)*0.01, f'{true_snr:.1f}',
                ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, scene_snr + max(overall_snrs)*0.01, f'{scene_snr:.1f}',
                ha='center', va='bottom', alpha=0.7)

    # Plot 2: Number of homogeneous regions found
    bars3 = ax2.bar(x, region_counts, color='blue', alpha=0.7)

    ax2.set_xlabel('Spectral Bands')
    ax2.set_ylabel('Number of Homogeneous Regions')
    ax2.set_title('Homogeneous Regions Detected')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add count labels
    for i, count in enumerate(region_counts):
        ax2.text(i, count + max(region_counts)*0.01, f'{count}', ha='center', va='bottom')

    # Plot 3: SNR distribution for each band
    for i, band in enumerate(bands):
        if 'individual_snrs' in snr_results[band] and snr_results[band]['individual_snrs']:
            snrs = snr_results[band]['individual_snrs']
            ax3.scatter([i] * len(snrs), snrs, alpha=0.6, s=20)

    ax3.set_xlabel('Spectral Bands')
    ax3.set_ylabel('Individual Region SNR')
    ax3.set_title('SNR Distribution Across Homogeneous Regions')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bands, rotation=45)
    ax3.grid(True, alpha=0.3)

    # Plot 4: SNR improvement factor
    snr_ratios = [true_snrs[i] / overall_snrs[i] if overall_snrs[i] > 0 else 0 for i in range(len(bands))]
    bars4 = ax4.bar(x, snr_ratios, color='purple', alpha=0.7)

    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
    ax4.set_xlabel('Spectral Bands')
    ax4.set_ylabel('SNR Improvement Factor')
    ax4.set_title('True SNR vs Scene Variation Ratio')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bands, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add ratio labels
    for i, ratio in enumerate(snr_ratios):
        ax4.text(i, ratio + max(snr_ratios)*0.01, f'{ratio:.1f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"SNR analysis plot saved: {output_filename}")
    plt.show()

def create_region_visualization(image_data, homogeneous_regions, band_name, max_regions=20):
    """
    Visualize detected homogeneous regions on the image.
    """
    if not homogeneous_regions:
        print(f"No homogeneous regions to visualize for {band_name}")
        return

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Display original image
    ax1.imshow(image_data, cmap='gray', vmin=0, vmax=np.percentile(image_data[image_data > 0], 99))
    ax1.set_title(f'{band_name} Band - Original Image')
    ax1.axis('off')

    # Display image with homogeneous regions highlighted
    ax2.imshow(image_data, cmap='gray', vmin=0, vmax=np.percentile(image_data[image_data > 0], 99))

    # Highlight homogeneous regions
    regions_to_show = homogeneous_regions[:max_regions]  # Limit for visibility
    colors = plt.cm.rainbow(np.linspace(0, 1, len(regions_to_show)))

    for i, region in enumerate(regions_to_show):
        x1, y1, x2, y2 = region['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                           edgecolor=colors[i], facecolor='none', alpha=0.8)
        ax2.add_patch(rect)

        # Add SNR label
        ax2.text(x1+2, y1+10, f'SNR:{region["snr"]:.1f}',
                color=colors[i], fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax2.set_title(f'{band_name} - Homogeneous Regions (SNR Analysis)')
    ax2.axis('off')

    plt.tight_layout()
    output_file = f"homogeneous_regions_{band_name.lower()}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Region visualization saved: {output_file}")
    plt.show()

def create_snr_violin_plot(snr_results, output_filename="snr_violin_homogeneous.png"):
    """
    Create violin plot showing SNR distribution of homogeneous regions for each band.
    """
    bands = list(snr_results.keys())
    if not bands:
        return

    # Prepare data for violin plot
    snr_data = []
    band_labels = []

    for band in bands:
        if 'individual_snrs' in snr_results[band] and snr_results[band]['individual_snrs']:
            snr_data.append(snr_results[band]['individual_snrs'])
            band_labels.append(band)

    if not snr_data:
        print("No individual SNR data available for violin plot")
        return

    # Create the violin plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create violin plot
    violins = ax.violinplot(snr_data, positions=range(len(band_labels)),
                           showmeans=True, showmedians=True, showextrema=True)

    # Customize violin plot appearance
    for pc in violins['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
        pc.set_edgecolor('navy')

    violins['cmeans'].set_color('red')
    violins['cmeans'].set_linewidth(2)
    violins['cmedians'].set_color('black')
    violins['cmedians'].set_linewidth(2)

    # Add individual data points as overlay
    for i, snrs in enumerate(snr_data):
        # Subsample for visibility if too many points
        if len(snrs) > 500:
            sample_size = 500
            indices = np.random.choice(len(snrs), sample_size, replace=False)
            snrs_sample = [snrs[j] for j in indices]
        else:
            snrs_sample = snrs

        # Add jitter to x-coordinates for visibility
        x_jitter = np.random.normal(i, 0.02, len(snrs_sample))
        ax.scatter(x_jitter, snrs_sample, alpha=0.3, s=8, color='darkblue')

    # Customize the plot
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels, rotation=45)
    ax.set_xlabel('Spectral Bands')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('SNR Distribution of Homogeneous Regions\nViolin Plot with Individual Data Points',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = []
    for i, (band, snrs) in enumerate(zip(band_labels, snr_data)):
        mean_snr = np.mean(snrs)
        median_snr = np.median(snrs)
        std_snr = np.std(snrs)
        count = len(snrs)
        stats_text.append(f"{band}: μ={mean_snr:.1f}, σ={std_snr:.1f}, n={count}")

    # Add text box with statistics
    textstr = '\n'.join(stats_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"SNR violin plot saved: {output_filename}")
    plt.show()

def create_snr_spatial_maps(snr_results, output_filename="snr_spatial_maps.png"):
    """
    Create 2D spatial maps showing SNR distribution across the image for each band.
    """
    bands = list(snr_results.keys())
    if not bands:
        return

    # Determine image dimensions from the first band's regions
    first_band = bands[0]
    if 'homogeneous_regions' not in snr_results[first_band] or not snr_results[first_band]['homogeneous_regions']:
        print("No homogeneous regions data available for spatial mapping")
        return

    # Get image dimensions from bounding boxes
    all_regions = snr_results[first_band]['homogeneous_regions']
    max_x = max(region['bbox'][2] for region in all_regions)  # x2
    max_y = max(region['bbox'][3] for region in all_regions)  # y2

    print(f"Creating spatial SNR maps with dimensions: {max_x} x {max_y}")

    # Create subplots for each band
    n_bands = len(bands)
    cols = 3 if n_bands > 2 else n_bands
    rows = (n_bands + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_bands == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Spatial SNR Distribution Maps\nColor represents SNR values from homogeneous regions',
                 fontsize=16, fontweight='bold')

    for i, band in enumerate(bands):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # Get regions and SNR data for this band
        regions = snr_results[band]['homogeneous_regions']
        if not regions:
            ax.text(0.5, 0.5, f'No data for {band}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{band.title()} Band')
            continue

        # Create SNR map
        snr_map = np.full((max_y, max_x), np.nan)

        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            snr_val = region['snr']

            # Fill the region with its SNR value
            snr_map[y1:y2, x1:x2] = snr_val

        # Create the plot
        im = ax.imshow(snr_map, cmap='viridis', interpolation='nearest', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('SNR', rotation=270, labelpad=15)

        # Set title with statistics
        mean_snr = snr_results[band]['snr']
        region_count = len(regions)
        ax.set_title(f'{band.title()} Band\nMean SNR: {mean_snr:.1f} | Regions: {region_count}')

        # Set labels
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        # Add grid
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_bands, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"SNR spatial maps saved: {output_filename}")
    plt.show()

def main():
    """Main function for homogeneous region SNR analysis."""
    parser = argparse.ArgumentParser(description='SNR analysis using homogeneous regions')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_individual_enhanced_*.tiff')
    parser.add_argument('--detection-method', choices=['sliding_window', 'clustering'], default='sliding_window')
    parser.add_argument('--snr-method', choices=['median', 'mean', 'weighted_mean', 'pooled'], default='pooled')
    parser.add_argument('--compare-normalized', action='store_true',
                       help='Compare with 16-bit normalized versions')
    parser.add_argument('--visualize-regions', action='store_true',
                       help='Create region visualization plots')
    parser.add_argument('--window-size', type=int, default=32,
                       help='Size of analysis window in pixels (default: 32)')
    parser.add_argument('--stride', type=int, default=16,
                       help='Step size for sliding window (default: 16)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Homogeneity threshold (CV) for region detection (default: 0.1)')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Find input files
    current_dir = Path('.')
    band_files = {}
    expected_bands = ['green', 'nir', 'blue', 'red_edge', 'red']

    print(f"Signal-to-Noise Ratio Analysis from Homogeneous Regions")
    print(f"Detection method: {args.detection_method}")
    print(f"SNR calculation: {args.snr_method}")
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

    # Analyze original images
    print(f"\n1. Analyzing original images...")
    original_snr_results = {}

    for band_name, file_path in band_files.items():
        image_data = read_geotiff_10bit(file_path)
        snr_stats = analyze_band_snr(image_data, band_name, args.detection_method, args.snr_method,
                                    args.window_size, args.stride, args.threshold, args.verbose)
        original_snr_results[band_name] = snr_stats

        if args.visualize_regions:
            create_region_visualization(image_data, snr_stats['homogeneous_regions'], band_name)

    # Compare with normalized versions if requested
    normalized_snr_results = {}
    if args.compare_normalized:
        print(f"\n2. Analyzing 16-bit normalized images...")

        for band_name in band_files.keys():
            if band_name == 'green':
                continue  # Skip reference band

            normalized_file = f"pushbroom_aligned_{band_name}_236images_start0_25pxsec_greenpan_16bit_normalized.tiff"
            normalized_path = current_dir / normalized_file

            if normalized_path.exists():
                # Read as 16-bit format, convert to 10-bit equivalent for comparison
                with Image.open(normalized_path) as img:
                    normalized_data = np.array(img) / 16.0

                snr_stats = analyze_band_snr(normalized_data, f"{band_name}_normalized",
                                           args.detection_method, args.snr_method,
                                           args.window_size, args.stride, args.threshold, args.verbose)
                normalized_snr_results[band_name] = snr_stats
            else:
                print(f"    Normalized file not found: {normalized_file}")

    # Results summary
    print(f"\n3. SNR ANALYSIS SUMMARY:")
    print(f"   Method: Homogeneous regions ({args.detection_method} detection)")
    print(f"   SNR calculation: {args.snr_method}")
    print()

    print("Original Images:")
    for band_name, stats in original_snr_results.items():
        print(f"  {band_name:12}: SNR = {stats['snr']:6.2f} (from {stats['count']:3d} regions)")
        print(f"               Scene variation = {stats['overall_snr']:6.2f} (misleading)")

    if normalized_snr_results:
        print("\nAfter 16-bit Normalization:")
        for band_name, stats in normalized_snr_results.items():
            original_snr = original_snr_results[band_name]['snr']
            improvement = stats['snr'] / original_snr if original_snr > 0 else 0
            print(f"  {band_name:12}: SNR = {stats['snr']:6.2f} (from {stats['count']:3d} regions)")
            print(f"               Improvement = {improvement:6.2f}x")

    # Create visualizations
    print(f"\n4. Creating SNR analysis plots...")
    create_snr_analysis_plot(original_snr_results, "snr_original_homogeneous.png")
    create_snr_violin_plot(original_snr_results, "snr_violin_homogeneous.png")
    create_snr_spatial_maps(original_snr_results, "snr_spatial_maps.png")

    if normalized_snr_results:
        create_snr_analysis_plot(normalized_snr_results, "snr_normalized_homogeneous.png")
        create_snr_violin_plot(normalized_snr_results, "snr_violin_normalized.png")
        create_snr_spatial_maps(normalized_snr_results, "snr_spatial_maps_normalized.png")

if __name__ == "__main__":
    main()