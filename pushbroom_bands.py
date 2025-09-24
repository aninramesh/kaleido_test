#!/usr/bin/env python3
"""
Script to extract spectral bands and create pushbroom images from 10-bit GeoTIFF files.
Bands: NIR [6:114], Blue [114:222], Red-edge [222:330], Red [330:438], Green-pan [438:726]
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage

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

def extract_bands(image_data):
    """
    Extract the 5 spectral bands from the image data.
    
    Args:
        image_data (numpy.ndarray): Image data with shape (726, 2336)
        
    Returns:
        dict: Dictionary containing each band
    """
    bands = {}
    
    # Define band ranges (y-coordinates)
    band_ranges = {
        'nir': (6, 114),
        'blue': (114, 222), 
        'red_edge': (222, 330),
        'red': (330, 438),
        'green_pan': (438, 726)
    }
    
    for band_name, (start, end) in band_ranges.items():
        # Extract band (y-direction is first dimension)
        band_data = image_data[start:end, :]
        bands[band_name] = band_data
        print(f"  {band_name}: shape {band_data.shape}, range {band_data.min()}-{band_data.max()}")
    
    return bands


def calculate_autocorrelation_with_rmse(img1, img2, x_range=(-6, 6), y_range=(26, 39), exclude_edge_x=True, edge_x_width=32):
    """
    Calculate combined metric using both autocorrelation and RMSE for finding best shift.
    Positive y_shift means img2 is shifted UP relative to img1.
    Positive x_shift means img2 is shifted right relative to img1.
    
    Args:
        img1 (numpy.ndarray): First image (reference)
        img2 (numpy.ndarray): Second image to correlate (will be shifted)
        x_range (tuple): Range of x-axis shifts to test (min, max) - can be negative
        y_range (tuple): Range of y-axis shifts to test (min, max) - positive values
        exclude_edge_x (bool): Whether to exclude problematic edge regions from analysis (default: True)
        edge_x_width (int): Width of edge region to exclude from left side (default: 32)
        
    Returns:
        tuple: (best_x_shift, best_y_shift, max_correlation, min_rmse, best_combined_score)
    """
    print(f"    Calculating autocorrelation + RMSE...")
    print(f"    X shift range: {x_range[0]} to {x_range[1]} pixels")
    print(f"    Y shift range: {y_range[0]} to {y_range[1]} pixels")
    if exclude_edge_x:
        print(f"    Excluding edge region: x[0:{edge_x_width}] from analysis")
    
    # Convert to float for calculations
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    # Optionally exclude problematic edge region from analysis
    if exclude_edge_x:
        img1_float = img1_float[:, edge_x_width:]
        img2_float = img2_float[:, edge_x_width:]
        print(f"    Analysis region shape after edge exclusion: {img1_float.shape}")
    
    # Normalize images to have zero mean and unit variance for correlation
    img1_norm = (img1_float - np.mean(img1_float)) / np.std(img1_float)
    img2_norm = (img2_float - np.mean(img2_float)) / np.std(img2_float)
    
    # Initialize tracking
    x_shifts = range(x_range[0], x_range[1] + 1)
    y_shifts = range(y_range[0], y_range[1] + 1)
    
    correlations = []
    rmse_values = []
    shift_coords = []
    
    # Test each shift combination
    for y_shift in y_shifts:
        for x_shift in x_shifts:
            # Calculate overlapping regions when img2 is shifted by (x_shift, y_shift)
            y_overlap_start = max(0, -y_shift)
            y_overlap_end = min(img1.shape[0], img2.shape[0] - y_shift)
            x_overlap_start = max(0, x_shift)
            x_overlap_end = min(img1.shape[1], img2.shape[1] + x_shift)
            
            # Extract regions from img1 (reference frame)
            y1_start = y_overlap_start
            y1_end = y_overlap_end
            x1_start = x_overlap_start
            x1_end = x_overlap_end
            
            # Extract regions from img2 (shifted frame, UP by y_shift)
            y2_start = y_overlap_start + y_shift
            y2_end = y_overlap_end + y_shift
            x2_start = x_overlap_start - x_shift
            x2_end = x_overlap_end - x_shift
            
            # Extract overlapping regions
            if (y1_end > y1_start and x1_end > x1_start and 
                y2_end > y2_start and x2_end > x2_start):
                
                # For correlation: use normalized regions
                region1_norm = img1_norm[y1_start:y1_end, x1_start:x1_end]
                region2_norm = img2_norm[y2_start:y2_end, x2_start:x2_end]
                
                # For RMSE: use original intensity regions
                region1_orig = img1_float[y1_start:y1_end, x1_start:x1_end]
                region2_orig = img2_float[y2_start:y2_end, x2_start:x2_end]
                
                # Ensure regions have the same size
                min_h = min(region1_norm.shape[0], region2_norm.shape[0])
                min_w = min(region1_norm.shape[1], region2_norm.shape[1])
                
                region1_norm = region1_norm[:min_h, :min_w]
                region2_norm = region2_norm[:min_h, :min_w]
                region1_orig = region1_orig[:min_h, :min_w]
                region2_orig = region2_orig[:min_h, :min_w]
                
                # Calculate metrics if regions are valid
                if region1_norm.size > 0 and region2_norm.size > 0:
                    # Calculate normalized cross-correlation
                    mean1 = np.mean(region1_norm)
                    mean2 = np.mean(region2_norm)
                    std1 = np.std(region1_norm)
                    std2 = np.std(region2_norm)
                    
                    if std1 > 0 and std2 > 0:
                        correlation = np.mean((region1_norm - mean1) * (region2_norm - mean2)) / (std1 * std2)
                    else:
                        correlation = 0
                    
                    # Calculate RMSE on original intensity values
                    rmse = np.sqrt(np.mean((region1_orig - region2_orig) ** 2))
                    
                    correlations.append(correlation)
                    rmse_values.append(rmse)
                    shift_coords.append((x_shift, y_shift))
    
    # Normalize metrics for combination
    if len(correlations) > 0:
        correlations = np.array(correlations)
        rmse_values = np.array(rmse_values)
        
        # Normalize correlation to [0, 1] (correlation is already in [-1, 1])
        norm_correlation = (correlations + 1) / 2.0  # Now in [0, 1]
        
        # Normalize RMSE to [0, 1] and invert (lower RMSE is better)
        if rmse_values.max() > rmse_values.min():
            norm_rmse_inverted = 1.0 - (rmse_values - rmse_values.min()) / (rmse_values.max() - rmse_values.min())
        else:
            norm_rmse_inverted = np.ones_like(rmse_values)  # All RMSE values are the same
        
        # Combined score: weighted combination (you can adjust weights)
        alpha = 0.7  # Weight for correlation
        beta = 0.3   # Weight for RMSE
        combined_scores = alpha * norm_correlation + beta * norm_rmse_inverted
        
        # Find best shift
        best_idx = np.argmax(combined_scores)
        best_x_shift, best_y_shift = shift_coords[best_idx]
        best_correlation = correlations[best_idx]
        best_rmse = rmse_values[best_idx]
        best_combined_score = combined_scores[best_idx]
        
        print(f"    Best match: X={best_x_shift}, Y={best_y_shift}")
        print(f"    Correlation={best_correlation:.4f}, RMSE={best_rmse:.2f}, Combined={best_combined_score:.4f}")
        
        return best_x_shift, best_y_shift, best_correlation, best_rmse, best_combined_score
    else:
        print(f"    No valid overlapping regions found!")
        return 0, 0, 0, float('inf'), 0


def extract_new_pushbroom_data(band_data, x_shift, y_shift, reference_width):
    """
    Extract new pushbroom data using Y-shift to determine the new content region:
    1. Use y[0:y_shift] from the image (this is the new content not overlapping with previous image)
    2. Apply X-shift by spatially offsetting the extracted data in the output array

    Args:
        band_data (numpy.ndarray): Band data to extract from
        x_shift (int): Cumulative X-axis shift from reference (positive = shifted right)
        y_shift (int): Y-axis shift that defines how much new content is available
        reference_width (int): Width of the reference image to match

    Returns:
        numpy.ndarray: Extracted new data with spatial X-shift applied, padded to reference_width
    """
    print(f"      Extracting new data y[0:{y_shift}] with cumulative X-shift={x_shift}")

    # Extract y[0:y_shift] - this is the new content region
    start_y = 0
    end_y = min(y_shift, band_data.shape[0])

    # If y_shift is larger than image height, use all available rows
    if y_shift > band_data.shape[0]:
        print(f"      Warning: y_shift={y_shift} > image height={band_data.shape[0]}, using all available rows")
        end_y = band_data.shape[0]

    # Extract the full width new content region first
    extracted = band_data[start_y:end_y, :]
    print(f"      Extracted base region: y[{start_y}:{end_y}], x[0:{band_data.shape[1]}], shape={extracted.shape}")

    # Create output array with reference width, initialized to zeros
    output_height = extracted.shape[0]
    output_array = np.zeros((output_height, reference_width), dtype=extracted.dtype)

    # Apply spatial X-shift by placing the extracted data at the shifted position
    if x_shift >= 0:
        # Positive shift: place data shifted to the right
        start_col = x_shift
        end_col = min(start_col + extracted.shape[1], reference_width)
        source_end_col = end_col - start_col

        if start_col < reference_width:
            output_array[:, start_col:end_col] = extracted[:, :source_end_col]
            print(f"      Applied positive X-shift: placed data at x[{start_col}:{end_col}]")
        else:
            print(f"      Warning: X-shift={x_shift} places data completely outside reference width")
    else:
        # Negative shift: place data shifted to the left (crop left side of extracted data)
        source_start_col = -x_shift
        source_end_col = min(source_start_col + reference_width, extracted.shape[1])
        dest_end_col = source_end_col - source_start_col

        if source_start_col < extracted.shape[1]:
            output_array[:, :dest_end_col] = extracted[:, source_start_col:source_end_col]
            print(f"      Applied negative X-shift: used source x[{source_start_col}:{source_end_col}] -> dest x[0:{dest_end_col}]")
        else:
            print(f"      Warning: X-shift={x_shift} crops all data outside source width")

    print(f"      Final output shape: {output_array.shape}")

    return output_array


def create_pushbroom_image(band_data_list, band_name, fps_pixels=25):
    """
    Create a pushbroom image by stitching band images sequentially.
    - First image: use all pixels
    - Subsequent images: use only fps_pixels from the beginning (representing 1 second at 25 fps)
    - Each new frame is stitched to the end of the previous frame
    
    Args:
        band_data_list (list): List of band data arrays
        band_name (str): Name of the band
        fps_pixels (int): Number of pixels to use from each subsequent image
        
    Returns:
        numpy.ndarray: Stitched pushbroom image
    """
    if not band_data_list:
        return None
    
    # Start with the first image (all pixels)
    pushbroom = band_data_list[0].copy()
    
    # Add fps_pixels from each subsequent image to the end
    for i in range(1, len(band_data_list)):
        band_data = band_data_list[i]
        height = band_data.shape[0]
        
        if height >= fps_pixels:
            # Take fps_pixels from the beginning of the image
            selected_pixels = band_data[:fps_pixels, :]
        else:
            # If image is smaller than fps_pixels, use all pixels
            selected_pixels = band_data
        
        
        # Append to the end of the pushbroom
        pushbroom = np.vstack([selected_pixels, pushbroom])
    
    print(f"  {band_name} pushbroom: shape {pushbroom.shape} (first image: all pixels, others: {fps_pixels} pixels from start)")
    
    return pushbroom


def create_aligned_pushbroom_image(band_data_list, band_name, shifts_list, fps_pixels=25):
    """
    Create a pushbroom image by stitching band images with autocorrelation-based alignment.
    - First image: use all pixels (reference)
    - Subsequent images: use fps_pixels from aligned positions based on calculated shifts
    
    Args:
        band_data_list (list): List of band data arrays
        band_name (str): Name of the band
        shifts_list (list): List of (x_shift, y_shift) tuples for each image pair
        fps_pixels (int): Number of pixels to use from each subsequent image
        
    Returns:
        numpy.ndarray: Stitched aligned pushbroom image
    """
    if not band_data_list:
        return None
    
    # Start with the first image (all pixels, reference)
    pushbroom = band_data_list[0].copy()
    reference_width = band_data_list[0].shape[1]
    
    # Track cumulative shifts from the reference image
    cumulative_x_shift = 0
    cumulative_y_shift = 0
    
    # Add aligned pixels from each subsequent image
    for i in range(1, len(band_data_list)):
        if i-1 < len(shifts_list):  # Make sure we have shift data
            # Get shift for this image pair
            x_shift, y_shift = shifts_list[i-1]
            
            # Accumulate shifts from reference
            cumulative_x_shift += x_shift
            cumulative_y_shift += y_shift
            
            print(f"    Image {i+1}: applying cumulative shift X={cumulative_x_shift}, Y={cumulative_y_shift}")
            
            # Extract new pushbroom data using Y-shift to define new content region
            band_data = band_data_list[i]
            new_data = extract_new_pushbroom_data(
                band_data, cumulative_x_shift, y_shift, reference_width
            )
            
            # Append to the beginning of the pushbroom (newer images at top)
            pushbroom = np.vstack([new_data, pushbroom])
        else:
            print(f"    Image {i+1}: no shift data available, using original method")
            # Fallback to original method if no shift data
            band_data = band_data_list[i]
            if band_data.shape[0] >= fps_pixels:
                selected_pixels = band_data[:fps_pixels, :]
            else:
                selected_pixels = band_data
            pushbroom = np.vstack([selected_pixels, pushbroom])
    
    print(f"  {band_name} aligned pushbroom: shape {pushbroom.shape} (with autocorrelation alignment)")
    
    return pushbroom


def create_shift_metrics_plot(shifts_list, correlations_list, rmse_list, combined_scores_list, num_images, start_image=0):
    """
    Create a visualization showing shift parameters and metrics as a function of image index.
    
    Args:
        shifts_list (list): List of (x_shift, y_shift) tuples
        correlations_list (list): List of correlation values
        rmse_list (list): List of RMSE values
        combined_scores_list (list): List of combined scores
        num_images (int): Total number of images processed
    """
    print(f"\nCreating shift metrics visualization...")
    
    if len(shifts_list) == 0:
        print("  No shift data to plot")
        return
    
    # Extract data for plotting
    x_shifts = [shift[0] for shift in shifts_list]
    y_shifts = [shift[1] for shift in shifts_list]
    correlations = correlations_list
    rmse_values = rmse_list
    combined_scores = combined_scores_list
    
    # Calculate cumulative X-shifts
    cumulative_x_shifts = []
    cumulative_x = 0
    for x_shift in x_shifts:
        cumulative_x += x_shift
        cumulative_x_shifts.append(cumulative_x)
    
    # Normalize RMSE for plotting (0-1 scale, inverted so higher is better)
    rmse_array = np.array(rmse_values)
    if rmse_array.max() > rmse_array.min():
        normalized_rmse = 1.0 - (rmse_array - rmse_array.min()) / (rmse_array.max() - rmse_array.min())
    else:
        normalized_rmse = np.ones_like(rmse_array)
    
    # Image pair indices (1->2, 2->3, etc.)
    pair_indices = list(range(1, len(shifts_list) + 1))
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Shift Metrics Analysis for {num_images} Images', fontsize=16, fontweight='bold')
    
    # Plot 1: X and Y shifts (individual and cumulative X)
    ax1.plot(pair_indices, x_shifts, 'bo-', label='X Shift (individual)', linewidth=2, markersize=8)
    ax1.plot(pair_indices, cumulative_x_shifts, 'co-', label='X Shift (cumulative)', linewidth=2, markersize=8)
    ax1.plot(pair_indices, y_shifts, 'ro-', label='Y Shift', linewidth=2, markersize=8)
    ax1.set_xlabel('Image Pair Index')
    ax1.set_ylabel('Shift (pixels)')
    ax1.set_title('X and Y Shifts Between Consecutive Images')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(pair_indices)
    
    # Plot 2: Correlation values
    ax2.plot(pair_indices, correlations, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Image Pair Index')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Cross-Correlation Between Consecutive Images')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(pair_indices)
    ax2.set_ylim([0.9, 1])
    
    # Add correlation values as text annotations
    for i, corr in enumerate(correlations):
        ax2.annotate(f'{corr:.3f}', (pair_indices[i], corr), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot 3: Normalized RMSE (inverted, so higher is better)
    ax3.plot(pair_indices, normalized_rmse, 'mo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Image Pair Index')
    ax3.set_ylabel('Normalized RMSE (inverted)')
    ax3.set_title('Normalized RMSE Between Consecutive Images')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(pair_indices)
    ax3.set_ylim([0, 1])
    
    # Add RMSE values as text annotations
    for i, (norm_rmse, raw_rmse) in enumerate(zip(normalized_rmse, rmse_values)):
        ax3.annotate(f'{raw_rmse:.1f}', (pair_indices[i], norm_rmse), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot 4: Combined scores
    ax4.plot(pair_indices, combined_scores, 'co-', linewidth=2, markersize=8)
    ax4.set_xlabel('Image Pair Index')
    ax4.set_ylabel('Combined Score')
    ax4.set_title('Combined Metric (0.7×Correlation + 0.3×Norm_RMSE)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(pair_indices)
    ax4.set_ylim([0.9, 1])
    
    # Add combined scores as text annotations
    for i, score in enumerate(combined_scores):
        ax4.annotate(f'{score:.3f}', (pair_indices[i], score), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"shift_metrics_analysis_{num_images}images_start{start_image}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {plot_filename}")
    
    # Show the plot
    plt.show()


def save_10bit_tiff(img_array, output_path):
    """
    Save 10-bit image data as TIFF preserving the 10-bit range.
    
    Args:
        img_array (numpy.ndarray): 10-bit image data (0-1023)
        output_path (str): Output file path
    """
    # Save as 16-bit TIFF to preserve 10-bit precision
    # Scale back to 16-bit range for storage (multiply by 16)
    img_16bit = (img_array * 16).astype(np.uint16)
    
    # Save as TIFF
    Image.fromarray(img_16bit).save(output_path)
    print(f"  Saved: {output_path}")



def main():
    """Main function to process images and create pushbroom bands."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create pushbroom images from 10-bit GeoTIFF files')
    parser.add_argument('--fps-pixels', type=int, default=25, 
                       help='Number of pixels per second (default: 25)')
    parser.add_argument('--num-images', type=int, default=3,
                       help='Number of images to process (default: 3)')
    parser.add_argument('--exclude-edge-x', action='store_true', default=True,
                       help='Exclude problematic edge region from correlation analysis (default: True)')
    parser.add_argument('--no-exclude-edge-x', dest='exclude_edge_x', action='store_false',
                       help='Include edge region in correlation analysis')
    parser.add_argument('--edge-x-width', type=int, default=32,
                       help='Width of edge region to exclude from left side (default: 32)')
    parser.add_argument('--start-image', type=int, default=0,
                       help='Starting image index (0-based) for pushbroom processing (default: 0)')

    args = parser.parse_args()
    
    # Configuration variables
    fps_pixels = args.fps_pixels
    num_images = args.num_images
    exclude_edge_x = args.exclude_edge_x
    edge_x_width = args.edge_x_width
    start_image = args.start_image
    
    # Path to the dataset
    dataset_path = Path("IPS_Dataset")
    
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        return
    
    # Get list of TIFF files
    tiff_files = sorted(list(dataset_path.glob("*.tiff")))
    
    if not tiff_files:
        print("No TIFF files found in the dataset directory")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    print(f"Configuration: {fps_pixels} pixels per second, processing {num_images} images, starting from index {start_image}")

    # Validate start_image parameter
    if start_image < 0:
        print(f"Error: start-image index cannot be negative. Got: {start_image}")
        return
    if start_image >= len(tiff_files):
        print(f"Error: start-image index {start_image} is beyond available files ({len(tiff_files)} files found)")
        return

    # Calculate the range of images to process
    end_image = min(start_image + num_images, len(tiff_files))
    actual_num_images = end_image - start_image

    print(f"Processing images {start_image} to {end_image-1} ({actual_num_images} images total) for pushbroom creation...")
    
    # Initialize band data storage
    band_data_lists = {
        'nir': [],
        'blue': [],
        'red_edge': [],
        'red': [],
        'green_pan': []
    }
    
    # Process each image in the specified range
    for i, tiff_file in enumerate(tiff_files[start_image:end_image]):
        image_index = start_image + i
        print(f"\nProcessing image {image_index+1}/{len(tiff_files)} (local {i+1}/{actual_num_images}): {tiff_file.name}")
        
        # Read the 10-bit GeoTIFF
        image_data = read_geotiff_10bit(tiff_file)
        
        if image_data is not None:
            print(f"  Image shape: {image_data.shape}")
            
            # Extract bands
            bands = extract_bands(image_data)
            
            # Store band data for pushbroom creation
            for band_name, band_data in bands.items():
                band_data_lists[band_name].append(band_data)
        else:
            print(f"  Failed to read {tiff_file.name}")
    
    # Calculate shifts between consecutive images using green_pan band
    print(f"\nCalculating autocorrelation shifts between consecutive images...")
    shifts_list = []
    correlations_list = []
    rmse_list = []
    combined_scores_list = []
    
    if len(band_data_lists['green_pan']) >= 2:
        for i in range(len(band_data_lists['green_pan']) - 1):
            print(f"\nCalculating shift between image {i+1} and {i+2}:")
            green_pan_1 = band_data_lists['green_pan'][i]
            green_pan_2 = band_data_lists['green_pan'][i+1]
            
            # Calculate autocorrelation + RMSE combined shift
            x_shift, y_shift, correlation, rmse, combined_score = calculate_autocorrelation_with_rmse(
                green_pan_1, green_pan_2, 
                x_range=(-7, 7), 
                y_range=(20, 39),
                exclude_edge_x=exclude_edge_x,
                edge_x_width=edge_x_width
            )
            
            shifts_list.append((x_shift, y_shift))
            correlations_list.append(correlation)
            rmse_list.append(rmse)
            combined_scores_list.append(combined_score)
            
            print(f"  Shift for pair {i+1}->{i+2}: X={x_shift}, Y={y_shift}")
            print(f"  Metrics: Corr={correlation:.4f}, RMSE={rmse:.2f}, Combined={combined_score:.4f}")
    
    print(f"\nCalculated {len(shifts_list)} shift pairs for {actual_num_images} images")
    
    # Create visualization of shift parameters and metrics
    if len(shifts_list) > 0:
        create_shift_metrics_plot(shifts_list, correlations_list, rmse_list, combined_scores_list, actual_num_images, start_image)
    
    # Create aligned pushbroom images for each band
    print(f"\nCreating aligned pushbroom images...")
    
    for band_name, band_data_list in band_data_lists.items():
        if band_data_list:
            print(f"\nCreating {band_name} aligned pushbroom...")
            
            # Create aligned pushbroom image using calculated shifts
            pushbroom = create_aligned_pushbroom_image(band_data_list, band_name, shifts_list, fps_pixels)
            
            if pushbroom is not None:
                # Save aligned pushbroom image
                output_path = f"pushbroom_aligned_{band_name}_{actual_num_images}images_start{start_image}_{fps_pixels}pxsec.tiff"
                save_10bit_tiff(pushbroom, output_path)

                # Also create non-aligned version for comparison
                print(f"  Creating comparison (non-aligned) {band_name} pushbroom...")
                pushbroom_original = create_pushbroom_image(band_data_list, band_name, fps_pixels)
                if pushbroom_original is not None:
                    output_path_orig = f"pushbroom_original_{band_name}_{actual_num_images}images_start{start_image}_{fps_pixels}pxsec.tiff"
                    save_10bit_tiff(pushbroom_original, output_path_orig)
                
                # Also save individual band from first image for reference
                if band_data_list:
                    single_band_path = f"single_{band_name}_band.tiff"
                    save_10bit_tiff(band_data_list[0], single_band_path)
        else:
            print(f"No data available for {band_name} band")
    
    print(f"\nPushbroom processing complete!")
    print(f"Created pushbroom images for {actual_num_images} images (starting from index {start_image}) using {fps_pixels} pixels per second")

if __name__ == "__main__":
    main()
