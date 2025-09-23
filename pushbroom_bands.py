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
            return img_array

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


def calculate_autocorrelation(img1, img2, x_range=(-6, 6), y_range=(26, 39)):
    """
    Calculate autocorrelation between two images within specified shift ranges.
    Positive y_shift means img2 is shifted UP relative to img1.
    Positive x_shift means img2 is shifted right relative to img1.
    
    Args:
        img1 (numpy.ndarray): First image (reference)
        img2 (numpy.ndarray): Second image to correlate (will be shifted)
        x_range (tuple): Range of x-axis shifts to test (min, max) - can be negative
        y_range (tuple): Range of y-axis shifts to test (min, max) - positive values
        
    Returns:
        tuple: (best_x_shift, best_y_shift, max_correlation)
    """
    print(f"    Calculating autocorrelation...")
    print(f"    X shift range: {x_range[0]} to {x_range[1]} pixels")
    print(f"    Y shift range: {y_range[0]} to {y_range[1]} pixels")
    
    # Convert to float for calculations
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    # Normalize images to have zero mean and unit variance
    img1_norm = (img1_float - np.mean(img1_float)) / np.std(img1_float)
    img2_norm = (img2_float - np.mean(img2_float)) / np.std(img2_float)
    
    # Initialize correlation tracking
    x_shifts = range(x_range[0], x_range[1] + 1)
    y_shifts = range(y_range[0], y_range[1] + 1)
    
    best_correlation = -np.inf
    best_x_shift = 0
    best_y_shift = 0
    
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
                
                region1 = img1_norm[y1_start:y1_end, x1_start:x1_end]
                region2 = img2_norm[y2_start:y2_end, x2_start:x2_end]
                
                # Ensure regions have the same size
                min_h = min(region1.shape[0], region2.shape[0])
                min_w = min(region1.shape[1], region2.shape[1])
                
                region1 = region1[:min_h, :min_w]
                region2 = region2[:min_h, :min_w]
                
                # Calculate normalized cross-correlation
                if region1.size > 0 and region2.size > 0:
                    mean1 = np.mean(region1)
                    mean2 = np.mean(region2)
                    std1 = np.std(region1)
                    std2 = np.std(region2)
                    
                    if std1 > 0 and std2 > 0:
                        correlation = np.mean((region1 - mean1) * (region2 - mean2)) / (std1 * std2)
                    else:
                        correlation = 0
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_x_shift = x_shift
                        best_y_shift = y_shift
    
    print(f"    Best match: X={best_x_shift}, Y={best_y_shift}, Correlation={best_correlation:.4f}")
    
    return best_x_shift, best_y_shift, best_correlation


def align_and_extract_pixels(band_data, x_shift, y_shift, fps_pixels, reference_width):
    """
    Extract fps_pixels from a band while applying the calculated shift for alignment.
    Ensures output width matches reference_width.
    
    Args:
        band_data (numpy.ndarray): Band data to extract from
        x_shift (int): X-axis shift in pixels (positive = shifted right)
        y_shift (int): Y-axis shift in pixels (positive = shifted up)
        fps_pixels (int): Number of pixels to extract
        reference_width (int): Width of the reference image to match
        
    Returns:
        numpy.ndarray: Extracted and aligned pixels with reference_width
    """
    # For y_shift: positive means img2 is shifted UP relative to reference
    # So we need to start extraction from a position that accounts for this shift
    start_y = max(0, y_shift)
    end_y = min(band_data.shape[0], start_y + fps_pixels)
    
    # For x_shift: ensure we maintain the same width as reference
    # Calculate the x-region that would align with the reference
    if x_shift >= 0:
        # Image shifted right, take left portion
        start_x = 0
        end_x = min(reference_width, band_data.shape[1])
    else:
        # Image shifted left, take right portion  
        start_x = min(-x_shift, band_data.shape[1] - reference_width)
        end_x = start_x + reference_width
        end_x = min(end_x, band_data.shape[1])
    
    # Extract the aligned region
    extracted = band_data[start_y:end_y, start_x:end_x]
    
    # Ensure width matches reference_width
    if extracted.shape[1] < reference_width:
        # Pad with zeros if needed
        padding = reference_width - extracted.shape[1]
        extracted = np.pad(extracted, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    elif extracted.shape[1] > reference_width:
        # Truncate if too wide
        extracted = extracted[:, :reference_width]
    
    # Ensure we have the right number of rows
    if extracted.shape[0] < fps_pixels:
        # If not enough pixels available, use what we have
        return extracted
    else:
        # Take exactly fps_pixels
        return extracted[:fps_pixels, :]


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
            
            # Extract aligned pixels using cumulative shift
            band_data = band_data_list[i]
            aligned_pixels = align_and_extract_pixels(
                band_data, cumulative_x_shift, cumulative_y_shift, fps_pixels, reference_width
            )
            
            # Append to the beginning of the pushbroom (newer images at top)
            pushbroom = np.vstack([aligned_pixels, pushbroom])
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
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to process (default: 10)')
    
    args = parser.parse_args()
    
    # Configuration variables
    fps_pixels = args.fps_pixels
    num_images = args.num_images
    
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
    print(f"Configuration: {fps_pixels} pixels per second, processing {num_images} images")
    
    # Process specified number of images
    num_images = min(num_images, len(tiff_files))
    print(f"Processing first {num_images} images for pushbroom creation...")
    
    # Initialize band data storage
    band_data_lists = {
        'nir': [],
        'blue': [],
        'red_edge': [],
        'red': [],
        'green_pan': []
    }
    
    # Process each image
    for i, tiff_file in enumerate(tiff_files[:num_images]):
        print(f"\nProcessing image {i+1}/{num_images}: {tiff_file.name}")
        
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
    
    if len(band_data_lists['green_pan']) >= 2:
        for i in range(len(band_data_lists['green_pan']) - 1):
            print(f"\nCalculating shift between image {i+1} and {i+2}:")
            green_pan_1 = band_data_lists['green_pan'][i]
            green_pan_2 = band_data_lists['green_pan'][i+1]
            
            # Calculate autocorrelation shift
            x_shift, y_shift, correlation = calculate_autocorrelation(
                green_pan_1, green_pan_2, 
                x_range=(-6, 6), 
                y_range=(26, 39)
            )
            
            shifts_list.append((x_shift, y_shift))
            print(f"  Shift for pair {i+1}->{i+2}: X={x_shift}, Y={y_shift}, Corr={correlation:.4f}")
    
    print(f"\nCalculated {len(shifts_list)} shift pairs for {num_images} images")
    
    # Create aligned pushbroom images for each band
    print(f"\nCreating aligned pushbroom images...")
    
    for band_name, band_data_list in band_data_lists.items():
        if band_data_list:
            print(f"\nCreating {band_name} aligned pushbroom...")
            
            # Create aligned pushbroom image using calculated shifts
            pushbroom = create_aligned_pushbroom_image(band_data_list, band_name, shifts_list, fps_pixels)
            
            if pushbroom is not None:
                # Save aligned pushbroom image
                output_path = f"pushbroom_aligned_{band_name}_{num_images}images_{fps_pixels}pxsec.tiff"
                save_10bit_tiff(pushbroom, output_path)
                
                # Also create non-aligned version for comparison
                print(f"  Creating comparison (non-aligned) {band_name} pushbroom...")
                pushbroom_original = create_pushbroom_image(band_data_list, band_name, fps_pixels)
                if pushbroom_original is not None:
                    output_path_orig = f"pushbroom_original_{band_name}_{num_images}images_{fps_pixels}pxsec.tiff"
                    save_10bit_tiff(pushbroom_original, output_path_orig)
                
                # Also save individual band from first image for reference
                if band_data_list:
                    single_band_path = f"single_{band_name}_band.tiff"
                    save_10bit_tiff(band_data_list[0], single_band_path)
        else:
            print(f"No data available for {band_name} band")
    
    print(f"\nPushbroom processing complete!")
    print(f"Created pushbroom images for {num_images} images using {fps_pixels} pixels per second")

if __name__ == "__main__":
    main()
