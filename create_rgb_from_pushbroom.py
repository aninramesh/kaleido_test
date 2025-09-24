#!/usr/bin/env python3
"""
Script to create RGB images from pushbroom TIFF files.
Reads aligned pushbroom TIFF files and creates RGB compositions using band range information.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def read_pushbroom_tiff(file_path):
    """
    Read a pushbroom TIFF file and return the image data.

    Args:
        file_path (str): Path to the TIFF file

    Returns:
        numpy.ndarray: Image data
    """
    try:
        with Image.open(file_path) as img:
            img_array = np.array(img)
            print(f"  Loaded {file_path.name}: shape {img_array.shape}")
            return img_array
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def extract_rgb_bands(pushbroom_data, band_ranges):
    """
    Extract RGB bands from pushbroom data using band range information.

    Args:
        pushbroom_data (dict): Dictionary containing band name -> image data
        band_ranges (dict): Dictionary containing band name -> (start, end) ranges

    Returns:
        tuple: (red_band, green_band, blue_band) as numpy arrays
    """
    # Use red, green_pan (as green), and blue bands for RGB
    red_band = pushbroom_data.get('red')
    green_band = pushbroom_data.get('green_pan')  # Using green_pan as green channel
    blue_band = pushbroom_data.get('blue')

    if red_band is None or green_band is None or blue_band is None:
        missing = []
        if red_band is None: missing.append('red')
        if green_band is None: missing.append('green_pan')
        if blue_band is None: missing.append('blue')
        raise ValueError(f"Missing required bands for RGB: {missing}")

    print(f"  RGB bands extracted:")
    print(f"    Red: {red_band.shape}, range {red_band.min()}-{red_band.max()}")
    print(f"    Green (green_pan): {green_band.shape}, range {green_band.min()}-{green_band.max()}")
    print(f"    Blue: {blue_band.shape}, range {blue_band.min()}-{blue_band.max()}")

    return red_band, green_band, blue_band

def normalize_band_for_display(band_data, percentile_clip=(2, 98)):
    """
    Normalize band data for display using percentile clipping.

    Args:
        band_data (numpy.ndarray): Band data
        percentile_clip (tuple): Lower and upper percentiles for clipping (default: (2, 98))

    Returns:
        numpy.ndarray: Normalized band data (0-255, uint8)
    """
    # Calculate percentile bounds
    low_val = np.percentile(band_data, percentile_clip[0])
    high_val = np.percentile(band_data, percentile_clip[1])

    # Clip and normalize to 0-1
    clipped = np.clip(band_data, low_val, high_val)
    normalized = (clipped - low_val) / (high_val - low_val)

    # Convert to 0-255 uint8
    normalized_uint8 = (normalized * 255).astype(np.uint8)

    return normalized_uint8

def create_rgb_image(red_band, green_band, blue_band, enhance_contrast=True, red_shift_param=396, green_shift_param=612):
    """
    Create RGB image from individual band data with proper spatial alignment.

    Applies shifts to align bands spatially:
    - Blue band: reference (no shift)
    - Red band: shift down by 396 pixels to align with blue
    - Green band: shift down by 612 pixels to align with blue

    Args:
        red_band (numpy.ndarray): Red channel data
        green_band (numpy.ndarray): Green channel data
        blue_band (numpy.ndarray): Blue channel data
        enhance_contrast (bool): Whether to apply contrast enhancement
        red_shift_param (int): Pixels to shift red band down (default: 396)
        green_shift_param (int): Pixels to shift green band down (default: 612)

    Returns:
        numpy.ndarray: RGB image (height, width, 3)
    """
    print(f"  Original band shapes:")
    print(f"    Red: {red_band.shape}")
    print(f"    Green: {green_band.shape}")
    print(f"    Blue: {blue_band.shape}")

    # Apply spatial shifts to align bands with blue as reference (customizable)
    red_shift = red_shift_param    # Red shifts down
    green_shift = green_shift_param  # Green shifts down

    print(f"  Applying spatial shifts for band alignment:")
    print(f"    Red: shifting down by {red_shift} pixels")
    print(f"    Green: shifting down by {green_shift} pixels")
    print(f"    Blue: reference (no shift)")

    # Apply shifts by padding and cropping
    # Red band: add padding at top, crop from bottom
    red_shifted = np.pad(red_band, ((red_shift, 0), (0, 0)), mode='constant', constant_values=0)

    # Green band: add padding at top, crop from bottom
    green_shifted = np.pad(green_band, ((green_shift, 0), (0, 0)), mode='constant', constant_values=0)

    # Blue band: no shift needed (reference)
    blue_shifted = blue_band

    print(f"  Shifted band shapes:")
    print(f"    Red: {red_shifted.shape}")
    print(f"    Green: {green_shifted.shape}")
    print(f"    Blue: {blue_shifted.shape}")

    # Find the minimum dimensions after shifting
    min_height = min(red_shifted.shape[0], green_shifted.shape[0], blue_shifted.shape[0])
    min_width = min(red_shifted.shape[1], green_shifted.shape[1], blue_shifted.shape[1])

    # Crop all bands to the same size from the top-left
    red_cropped = red_shifted[:min_height, :min_width]
    green_cropped = green_shifted[:min_height, :min_width]
    blue_cropped = blue_shifted[:min_height, :min_width]

    print(f"  Final aligned size: {min_height}x{min_width}")

    if enhance_contrast:
        # Normalize each band for better display
        red_norm = normalize_band_for_display(red_cropped)
        green_norm = normalize_band_for_display(green_cropped)
        blue_norm = normalize_band_for_display(blue_cropped)
    else:
        # Simple scaling to 0-255
        red_norm = ((red_cropped / red_cropped.max()) * 255).astype(np.uint8)
        green_norm = ((green_cropped / green_cropped.max()) * 255).astype(np.uint8)
        blue_norm = ((blue_cropped / blue_cropped.max()) * 255).astype(np.uint8)

    # Create mask for areas where all channels have valid data (non-zero)
    valid_mask = (red_cropped > 0) & (green_cropped > 0) & (blue_cropped > 0)

    # Find the bounding box of valid data
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        print(f"  Warning: No overlapping valid data found in all three channels!")
        rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
    else:
        # Find the bounds of valid data
        row_start = np.where(valid_rows)[0][0]
        row_end = np.where(valid_rows)[0][-1] + 1
        col_start = np.where(valid_cols)[0][0]
        col_end = np.where(valid_cols)[0][-1] + 1

        print(f"  Valid data region: rows {row_start}:{row_end}, cols {col_start}:{col_end}")

        # Crop to valid region only
        red_valid = red_norm[row_start:row_end, col_start:col_end]
        green_valid = green_norm[row_start:row_end, col_start:col_end]
        blue_valid = blue_norm[row_start:row_end, col_start:col_end]

        # Stack into RGB image
        rgb_image = np.stack([red_valid, green_valid, blue_valid], axis=2)

    print(f"  Created RGB image: {rgb_image.shape}")

    return rgb_image

def save_rgb_image(rgb_image, output_path):
    """
    Save RGB image to file.

    Args:
        rgb_image (numpy.ndarray): RGB image data
        output_path (str): Output file path
    """
    rgb_pil = Image.fromarray(rgb_image)
    rgb_pil.save(output_path)
    print(f"  Saved RGB image: {output_path}")

def main():
    """Main function to create RGB images from pushbroom TIFF files."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create RGB images from pushbroom TIFF files')
    parser.add_argument('--input-pattern', type=str, default='pushbroom_aligned_*_*.tiff',
                       help='Input file pattern for pushbroom TIFF files (default: pushbroom_aligned_*_*.tiff)')
    parser.add_argument('--enhance-contrast', action='store_true', default=True,
                       help='Apply contrast enhancement using percentile clipping (default: True)')
    parser.add_argument('--no-enhance-contrast', dest='enhance_contrast', action='store_false',
                       help='Disable contrast enhancement')
    parser.add_argument('--red-shift', type=int, default=396,
                       help='Pixels to shift red band down for alignment with blue (default: 396)')
    parser.add_argument('--green-shift', type=int, default=612,
                       help='Pixels to shift green band down for alignment with blue (default: 612)')

    args = parser.parse_args()

    # Band ranges (matching the original script)
    band_ranges = {
        'nir': (6, 114),
        'blue': (114, 222),
        'red_edge': (222, 330),
        'red': (330, 438),
        'green_pan': (438, 726)
    }

    print(f"RGB Creation from Pushbroom TIFF Files")
    print(f"Band ranges: {band_ranges}")
    print(f"Contrast enhancement: {args.enhance_contrast}")

    # Find all pushbroom TIFF files
    current_dir = Path(".")
    tiff_files = list(current_dir.glob(args.input_pattern))

    if not tiff_files:
        print(f"No TIFF files found matching pattern: {args.input_pattern}")
        return

    print(f"\nFound {len(tiff_files)} TIFF files:")
    for tiff_file in tiff_files:
        print(f"  {tiff_file.name}")

    # Group files by band
    band_files = {}
    for tiff_file in tiff_files:
        filename = tiff_file.name
        for band_name in band_ranges.keys():
            if band_name in filename:
                band_files[band_name] = tiff_file
                break

    print(f"\nBand files identified:")
    for band_name, file_path in band_files.items():
        print(f"  {band_name}: {file_path.name}")

    # Check if we have the required bands for RGB
    required_bands = ['red', 'green_pan', 'blue']
    missing_bands = [band for band in required_bands if band not in band_files]
    if missing_bands:
        print(f"Error: Missing required bands for RGB: {missing_bands}")
        return

    # Load pushbroom data
    print(f"\nLoading pushbroom data...")
    pushbroom_data = {}
    for band_name, file_path in band_files.items():
        if band_name in required_bands:
            data = read_pushbroom_tiff(file_path)
            if data is not None:
                pushbroom_data[band_name] = data

    if len(pushbroom_data) != len(required_bands):
        print(f"Error: Failed to load all required bands")
        return

    # Extract RGB bands
    print(f"\nExtracting RGB bands...")
    red_band, green_band, blue_band = extract_rgb_bands(pushbroom_data, band_ranges)

    # Create RGB image
    print(f"\nCreating RGB image...")
    rgb_image = create_rgb_image(red_band, green_band, blue_band, args.enhance_contrast, args.red_shift, args.green_shift)

    # Save RGB image
    contrast_suffix = "_enhanced" if args.enhance_contrast else "_simple"
    output_filename = f"pushbroom_rgb{contrast_suffix}.png"
    save_rgb_image(rgb_image, output_filename)

    print(f"\nRGB creation complete!")
    print(f"Output saved as: {output_filename}")

if __name__ == "__main__":
    main()