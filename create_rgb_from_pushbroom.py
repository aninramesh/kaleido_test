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
from scipy import ndimage
import rasterio
from rasterio.transform import from_bounds

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

def normalize_band_for_display(band_data, percentile_clip=(2, 98), bits=8):
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
    if bits == 8:
        normalized_uint8 = (normalized * 255).astype(np.uint8)
    elif bits == 10:
        normalized_uint8 = (normalized * 1023).astype(np.uint16)
    else:
        raise ValueError("Unsupported bit depth. Use 8 or 10.")

    return normalized_uint8

def calculate_rgb_alignment(red_band, green_band, blue_band, x_range=(-4, 4), y_range=(-4, 4)):
    """
    Calculate optimal X and Y shifts for red and green bands to align with blue band using autocorrelation.
    Blue band is used as reference.

    Args:
        red_band (numpy.ndarray): Red channel data
        green_band (numpy.ndarray): Green channel data
        blue_band (numpy.ndarray): Blue channel data (reference)
        x_range (tuple): Range of x-axis shifts to test (min, max)
        y_range (tuple): Range of y-axis shifts to test (min, max)

    Returns:
        tuple: (red_x_shift, red_y_shift, green_x_shift, green_y_shift, red_correlation, green_correlation)
    """
    print(f"  Calculating RGB channel alignment using blue as reference...")
    print(f"  Search range: X={x_range[0]} to {x_range[1]}, Y={y_range[0]} to {y_range[1]} pixels")

    # Convert to float for calculations
    red_float = red_band.astype(np.float32)
    green_float = green_band.astype(np.float32)
    blue_float = blue_band.astype(np.float32)

    # Normalize all bands to 0-1023 range before autocorrelation
    def normalize_to_10bit(band_data, band_name):
        """Normalize band data to 0-1023 range"""
        min_val = band_data.min()
        max_val = band_data.max()
        if max_val > min_val:
            normalized = ((band_data - min_val) / (max_val - min_val)) * 1023.0
        else:
            normalized = np.zeros_like(band_data)
        print(f"    {band_name}: normalized from [{min_val:.1f}, {max_val:.1f}] to [0, 1023]")
        return normalized

    red_10bit = normalize_to_10bit(red_float, "Red")
    green_10bit = normalize_to_10bit(green_float, "Green")
    blue_10bit = normalize_to_10bit(blue_float, "Blue")

    # Normalize bands for correlation calculation (zero mean, unit variance)
    red_norm = (red_10bit - np.mean(red_10bit)) / np.std(red_10bit)
    green_norm = (green_10bit - np.mean(green_10bit)) / np.std(green_10bit)
    blue_norm = (blue_10bit - np.mean(blue_10bit)) / np.std(blue_10bit)

    def find_best_alignment(band_norm, band_name):
        """Find best alignment for a band against blue reference"""
        print(f"    Finding alignment for {band_name} band...")

        best_correlation = -1
        best_x_shift = 0
        best_y_shift = 0

        x_shifts = range(x_range[0], x_range[1] + 1)
        y_shifts = range(y_range[0], y_range[1] + 1)

        for y_shift in y_shifts:
            for x_shift in x_shifts:
                # Calculate overlapping region when band is shifted by (x_shift, y_shift)
                y_start = max(0, y_shift)
                y_end = min(blue_norm.shape[0], band_norm.shape[0] + y_shift)
                x_start = max(0, x_shift)
                x_end = min(blue_norm.shape[1], band_norm.shape[1] + x_shift)

                # Extract regions from blue (reference)
                blue_region = blue_norm[y_start:y_end, x_start:x_end]

                # Extract regions from shifted band
                band_y_start = y_start - y_shift
                band_y_end = y_end - y_shift
                band_x_start = x_start - x_shift
                band_x_end = x_end - x_shift

                if (band_y_end > band_y_start and band_x_end > band_x_start and
                    blue_region.size > 0):

                    band_region = band_norm[band_y_start:band_y_end, band_x_start:band_x_end]

                    # Ensure regions have same size
                    min_h = min(blue_region.shape[0], band_region.shape[0])
                    min_w = min(blue_region.shape[1], band_region.shape[1])

                    blue_region = blue_region[:min_h, :min_w]
                    band_region = band_region[:min_h, :min_w]

                    if blue_region.size > 0 and band_region.size > 0:
                        # Calculate normalized cross-correlation
                        blue_mean = np.mean(blue_region)
                        band_mean = np.mean(band_region)
                        blue_std = np.std(blue_region)
                        band_std = np.std(band_region)

                        if blue_std > 0 and band_std > 0:
                            correlation = np.mean((blue_region - blue_mean) * (band_region - band_mean)) / (blue_std * band_std)

                            if correlation > best_correlation:
                                best_correlation = correlation
                                best_x_shift = x_shift
                                best_y_shift = y_shift

        print(f"      Best {band_name} alignment: X={best_x_shift}, Y={best_y_shift}, Correlation={best_correlation:.4f}")
        return best_x_shift, best_y_shift, best_correlation

    # Find best alignment for red and green bands
    red_x_shift, red_y_shift, red_correlation = find_best_alignment(red_norm, "red")
    green_x_shift, green_y_shift, green_correlation = find_best_alignment(green_norm, "green")

    return red_x_shift, red_y_shift, green_x_shift, green_y_shift, red_correlation, green_correlation

def create_rgb_image(red_band, green_band, blue_band, enhance_contrast=True, red_shift_param=396, green_shift_param=612, red_x_shift=0, green_x_shift=0, blue_x_shift=0):
    """
    Create RGB image from individual band data with proper spatial alignment.

    Applies shifts to align bands spatially:
    - Blue band: reference (customizable x_shift)
    - Red band: shift down by 396 pixels and horizontally by red_x_shift to align with blue
    - Green band: shift down by 612 pixels and horizontally by green_x_shift to align with blue

    Args:
        red_band (numpy.ndarray): Red channel data
        green_band (numpy.ndarray): Green channel data
        blue_band (numpy.ndarray): Blue channel data
        enhance_contrast (bool): Whether to apply contrast enhancement
        red_shift_param (int): Pixels to shift red band down (default: 396)
        green_shift_param (int): Pixels to shift green band down (default: 612)
        red_x_shift (int): Pixels to shift red band horizontally (positive=right, negative=left, default: 0)
        green_x_shift (int): Pixels to shift green band horizontally (positive=right, negative=left, default: 0)
        blue_x_shift (int): Pixels to shift blue band horizontally (positive=right, negative=left, default: 0)

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
    print(f"    Red: shifting down by {red_shift} pixels, horizontally by {red_x_shift} pixels")
    print(f"    Green: shifting down by {green_shift} pixels, horizontally by {green_x_shift} pixels")
    print(f"    Blue: horizontally by {blue_x_shift} pixels")

    # Apply Y-shifts (vertical) by padding and cropping
    # Red band: add padding at top, crop from bottom
    red_y_shifted = np.pad(red_band, ((red_shift, 0), (0, 0)), mode='constant', constant_values=0)

    # Green band: add padding at top, crop from bottom
    green_y_shifted = np.pad(green_band, ((green_shift, 0), (0, 0)), mode='constant', constant_values=0)

    # Blue band: no Y-shift needed
    blue_y_shifted = blue_band

    # Apply X-shifts (horizontal) by padding and cropping
    def apply_x_shift(band_data, x_shift):
        if x_shift == 0:
            return band_data
        elif x_shift > 0:
            # Shift right: add padding on left
            return np.pad(band_data, ((0, 0), (x_shift, 0)), mode='constant', constant_values=0)
        else:
            # Shift left: crop from left and add padding on right
            abs_shift = -x_shift
            if abs_shift >= band_data.shape[1]:
                # Shift is larger than width, return all zeros
                return np.zeros_like(band_data)
            else:
                # Crop from left and add padding on right
                cropped = band_data[:, abs_shift:]
                return np.pad(cropped, ((0, 0), (0, abs_shift)), mode='constant', constant_values=0)

    # Apply X-shifts to all bands
    red_shifted = apply_x_shift(red_y_shifted, red_x_shift)
    green_shifted = apply_x_shift(green_y_shifted, green_x_shift)
    blue_shifted = apply_x_shift(blue_y_shifted, blue_x_shift)

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

def save_rgb_geotiff(rgb_image, output_path, transform=None, crs='EPSG:4326'):
    """
    Save RGB image as GeoTIFF file with proper georeferencing in 10-bit format.

    Args:
        rgb_image (numpy.ndarray): RGB image data (height, width, 3)
        output_path (str): Output GeoTIFF file path
        transform (rasterio.transform.Affine, optional): Geospatial transform
        crs (str): Coordinate reference system (default: 'EPSG:4326')
    """
    height, width, bands = rgb_image.shape

    # Convert to 10-bit (0-1023 range)
    rgb_10bit = (rgb_image.astype(np.float32) / 255.0 * 1023.0).astype(np.uint16)

    # Create default transform if none provided (basic pixel coordinates)
    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)

    # Save as GeoTIFF with RGB bands in 10-bit
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype='uint16',
        crs=crs,
        transform=transform,
        photometric='RGB',
        compress='lzw'
    ) as dst:
        # Write each RGB band
        for band_idx in range(bands):
            dst.write(rgb_10bit[:, :, band_idx], band_idx + 1)

    print(f"  Saved RGB GeoTIFF (10-bit): {output_path}")

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
    parser.add_argument('--red-x-shift', type=int, default=0,
                       help='Pixels to shift red band horizontally (positive=right, negative=left, default: 0)')
    parser.add_argument('--green-x-shift', type=int, default=0,
                       help='Pixels to shift green band horizontally (positive=right, negative=left, default: 0)')
    parser.add_argument('--blue-x-shift', type=int, default=0,
                       help='Pixels to shift blue band horizontally (positive=right, negative=left, default: 0)')
    parser.add_argument('--rgb-align-x-range', type=int, nargs=2, default=[-4, 4],
                       help='X-axis range for RGB autocorrelation alignment (default: -4 4)')
    parser.add_argument('--rgb-align-y-range', type=int, nargs=2, default=[-4, 4],
                       help='Y-axis range for RGB autocorrelation alignment (default: -4 4)')
    parser.add_argument('--disable-rgb-align', action='store_true', default=False,
                       help='Disable automatic RGB channel alignment using autocorrelation (default: False)')

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

    # Stage 1: Apply manual shifts first (coarse positioning)
    print(f"\nStage 1: Applying manual shifts for coarse band positioning...")
    print(f"  Manual shifts: Red Y={args.red_shift}, Green Y={args.green_shift}")

    # Apply manual shifts using existing create_rgb_image logic but extract intermediate results
    def apply_manual_shifts_to_bands(red_band, green_band, blue_band):
        """Apply manual Y and X shifts to get roughly aligned bands"""
        print(f"  Applying manual Y-shifts...")

        # Apply Y-shifts (vertical) - same logic as in create_rgb_image
        red_y_shifted = np.pad(red_band, ((args.red_shift, 0), (0, 0)), mode='constant', constant_values=0)
        green_y_shifted = np.pad(green_band, ((args.green_shift, 0), (0, 0)), mode='constant', constant_values=0)
        blue_y_shifted = blue_band  # Blue is reference for Y

        # Apply manual X-shifts if any
        def apply_manual_x_shift(band_data, x_shift):
            if x_shift == 0:
                return band_data
            elif x_shift > 0:
                return np.pad(band_data, ((0, 0), (x_shift, 0)), mode='constant', constant_values=0)
            else:
                abs_shift = -x_shift
                if abs_shift >= band_data.shape[1]:
                    return np.zeros_like(band_data)
                else:
                    cropped = band_data[:, abs_shift:]
                    return np.pad(cropped, ((0, 0), (0, abs_shift)), mode='constant', constant_values=0)

        red_shifted = apply_manual_x_shift(red_y_shifted, args.red_x_shift)
        green_shifted = apply_manual_x_shift(green_y_shifted, args.green_x_shift)
        blue_shifted = apply_manual_x_shift(blue_y_shifted, args.blue_x_shift)

        # Crop to same size
        min_height = min(red_shifted.shape[0], green_shifted.shape[0], blue_shifted.shape[0])
        min_width = min(red_shifted.shape[1], green_shifted.shape[1], blue_shifted.shape[1])

        red_cropped = red_shifted[:min_height, :min_width]
        green_cropped = green_shifted[:min_height, :min_width]
        blue_cropped = blue_shifted[:min_height, :min_width]

        print(f"    Manual alignment result: {min_height}x{min_width}")
        return red_cropped, green_cropped, blue_cropped

    # Apply manual shifts first
    manually_shifted_red, manually_shifted_green, manually_shifted_blue = apply_manual_shifts_to_bands(red_band, green_band, blue_band)

    # Stage 2: Apply RGB autocorrelation alignment (fine-tuning)
    if not args.disable_rgb_align:
        print(f"\nStage 2: Fine-tuning RGB channel alignment using autocorrelation...")
        print(f"  Search ranges: X={args.rgb_align_x_range}, Y={args.rgb_align_y_range}")

        rgb_red_x_shift, rgb_red_y_shift, rgb_green_x_shift, rgb_green_y_shift, red_correlation, green_correlation = calculate_rgb_alignment(
            manually_shifted_red, manually_shifted_green, manually_shifted_blue,
            x_range=tuple(args.rgb_align_x_range),
            y_range=tuple(args.rgb_align_y_range)
        )

        print(f"  Fine alignment results:")
        print(f"    Red band: X={rgb_red_x_shift}, Y={rgb_red_y_shift}, Correlation={red_correlation:.4f}")
        print(f"    Green band: X={rgb_green_x_shift}, Y={rgb_green_y_shift}, Correlation={green_correlation:.4f}")
        print(f"    Blue band: reference (no fine adjustment)")

        # Apply fine alignment shifts
        def apply_fine_shift(band_data, x_shift, y_shift, band_name):
            """Apply fine X and Y shifts for autocorrelation alignment"""
            print(f"    Applying {band_name} fine shift: X={x_shift}, Y={y_shift}")

            if x_shift == 0 and y_shift == 0:
                return band_data

            shifted_band = np.zeros_like(band_data)

            # Calculate regions with proper boundary handling
            src_y_start = max(0, -y_shift)
            src_y_end = min(band_data.shape[0], band_data.shape[0] - y_shift)
            src_x_start = max(0, -x_shift)
            src_x_end = min(band_data.shape[1], band_data.shape[1] - x_shift)

            dst_y_start = max(0, y_shift)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            dst_x_start = max(0, x_shift)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)

            if (src_y_end > src_y_start and src_x_end > src_x_start and
                dst_y_end <= shifted_band.shape[0] and dst_x_end <= shifted_band.shape[1]):
                shifted_band[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    band_data[src_y_start:src_y_end, src_x_start:src_x_end]

            return shifted_band

        # Apply fine alignment to manually shifted bands
        final_red_band = apply_fine_shift(manually_shifted_red, rgb_red_x_shift, rgb_red_y_shift, "Red")
        final_green_band = apply_fine_shift(manually_shifted_green, rgb_green_x_shift, rgb_green_y_shift, "Green")
        final_blue_band = manually_shifted_blue  # Blue is reference

        print(f"  RGB bands are now optimally aligned (manual + autocorrelation)")
    else:
        print(f"\nStage 2: Autocorrelation alignment disabled, using manual alignment only...")
        final_red_band = manually_shifted_red
        final_green_band = manually_shifted_green
        final_blue_band = manually_shifted_blue

    # Create RGB image from aligned bands (no additional shifts needed)
    print(f"\nCreating RGB image from aligned bands...")

    # Find valid data region
    valid_mask = (final_red_band > 0) & (final_green_band > 0) & (final_blue_band > 0)
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        print(f"  Warning: No overlapping valid data found in all three channels!")
        rgb_image = np.stack([
            normalize_band_for_display(final_red_band),
            normalize_band_for_display(final_green_band),
            normalize_band_for_display(final_blue_band)
        ], axis=2)
    else:
        row_start = np.where(valid_rows)[0][0]
        row_end = np.where(valid_rows)[0][-1] + 1
        col_start = np.where(valid_cols)[0][0]
        col_end = np.where(valid_cols)[0][-1] + 1

        print(f"  Valid RGB region: rows {row_start}:{row_end}, cols {col_start}:{col_end}")

        red_valid = final_red_band[row_start:row_end, col_start:col_end]
        green_valid = final_green_band[row_start:row_end, col_start:col_end]
        blue_valid = final_blue_band[row_start:row_end, col_start:col_end]

        # Apply contrast enhancement
        if args.enhance_contrast:
            red_norm = normalize_band_for_display(red_valid)
            green_norm = normalize_band_for_display(green_valid)
            blue_norm = normalize_band_for_display(blue_valid)
        else:
            red_norm = ((red_valid / red_valid.max()) * 255).astype(np.uint8)
            green_norm = ((green_valid / green_valid.max()) * 255).astype(np.uint8)
            blue_norm = ((blue_valid / blue_valid.max()) * 255).astype(np.uint8)

        rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)

    print(f"  Final RGB image: {rgb_image.shape}")

    # Save RGB image as PNG
    contrast_suffix = "_enhanced" if args.enhance_contrast else "_simple"
    output_filename_png = f"pushbroom_rgb{contrast_suffix}.png"
    save_rgb_image(rgb_image, output_filename_png)

    # Save RGB image as GeoTIFF
    output_filename_tiff = f"pushbroom_rgb{contrast_suffix}.tiff"
    save_rgb_geotiff(rgb_image, output_filename_tiff)

    print(f"\nRGB creation complete!")
    print(f"PNG output saved as: {output_filename_png}")
    print(f"GeoTIFF output saved as: {output_filename_tiff}")

if __name__ == "__main__":
    main()