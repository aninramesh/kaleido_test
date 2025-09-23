#!/usr/bin/env python3
"""
Simple script to read a single 10-bit GeoTIFF file.
Handles 10-bit data that may be scaled by 16 in the GeoTIFF format.
"""

import numpy as np
from PIL import Image

def read_geotiff_10bit(file_path):
    """
    Simple function to read a 10-bit GeoTIFF file.
    
    Args:
        file_path (str): Path to the GeoTIFF file
        
    Returns:
        numpy.ndarray: Image data in true 10-bit range (0-1023)
    """
    # Open and read the GeoTIFF file
    with Image.open(file_path) as img:
        # Convert to numpy array
        img_array = np.array(img)
        
        print(f"File: {file_path}")
        print(f"Shape: {img_array.shape}")
        print(f"Data type: {img_array.dtype}")
        print(f"Raw value range: {img_array.min()} - {img_array.max()}")
        
        # Check if it's 10-bit data scaled by 16 (common in GeoTIFF)
        max_val = img_array.max()
        min_val = img_array.min()
        range_val = max_val - min_val
        
        if range_val <= 1023 * 16:  # 10-bit scaled by 16
            # Convert back to true 10-bit range (0-1023)
            true_10bit = img_array / 16.0
            print(f"Detected 10-bit GeoTIFF scaled by 16")
            print(f"True 10-bit range: {true_10bit.min():.1f} - {true_10bit.max():.1f}")
            return true_10bit.astype(np.uint16)
        elif max_val <= 1023:
            # True 10-bit data
            print(f"Detected true 10-bit data")
            return img_array
        else:
            print(f"Warning: Data doesn't appear to be 10-bit format")
            return img_array

def save_as_10bit_tiff(img_array, output_path="converted_10bit.tiff"):
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
    print(f"Saved 10-bit data as '{output_path}' (stored as 16-bit TIFF)")

# Example usage
if __name__ == "__main__":
    # Read the specific GeoTIFF file you have open
    file_path = "IPS_Dataset/image_003711.tiff"
    
    try:
        # Read the 10-bit GeoTIFF
        image_data = read_geotiff_10bit(file_path)
        print(f"\nSuccessfully loaded image with shape {image_data.shape}")
        print(f"Final 10-bit range: {image_data.min()} - {image_data.max()}")
        
        # Save as 10-bit TIFF preserving the original precision
        save_as_10bit_tiff(image_data)
        
    except Exception as e:
        print(f"Error: {e}")