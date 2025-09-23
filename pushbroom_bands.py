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
    
    # Create pushbroom images for each band
    print(f"\nCreating pushbroom images...")
    
    for band_name, band_data_list in band_data_lists.items():
        if band_data_list:
            print(f"\nCreating {band_name} pushbroom...")
            
            # Create pushbroom image with fps_pixels parameter
            pushbroom = create_pushbroom_image(band_data_list, band_name, fps_pixels)
            
            if pushbroom is not None:
                # Save pushbroom image
                output_path = f"pushbroom_{band_name}_{num_images}images_{fps_pixels}pxsec.tiff"
                save_10bit_tiff(pushbroom, output_path)
                
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
