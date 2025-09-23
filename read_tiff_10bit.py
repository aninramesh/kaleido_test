#!/usr/bin/env python3
"""
Script to read 10-bit TIFF files from the IPS_Dataset directory.
This script handles 10-bit TIFF images properly by preserving the full bit depth.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def read_10bit_geotiff(file_path):
    """
    Read a 10-bit GeoTIFF file and return the image data in true 10-bit range.
    
    Args:
        file_path (str): Path to the GeoTIFF file
        
    Returns:
        numpy.ndarray: Image data in true 10-bit range (0-1023)
    """
    try:
        # Open the GeoTIFF file
        with Image.open(file_path) as img:
            # Convert to numpy array
            img_array = np.array(img)
            
            # Get image properties
            print(f"Image shape: {img_array.shape}")
            print(f"Data type: {img_array.dtype}")
            print(f"Raw min value: {img_array.min()}")
            print(f"Raw max value: {img_array.max()}")
            
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
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def display_image(img_array, title="10-bit GeoTIFF Image"):
    """
    Display the image using matplotlib.
    
    Args:
        img_array (numpy.ndarray): 10-bit image data (0-1023)
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(img_array, cmap='gray', vmin=0, vmax=1023)
    plt.colorbar(label='10-bit Pixel Value (0-1023)')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_as_10bit_tiff(img_array, output_path):
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
    print(f"10-bit data saved as: {output_path} (stored as 16-bit TIFF)")

def analyze_image_properties(img_array):
    """
    Analyze and print detailed properties of the image.
    
    Args:
        img_array (numpy.ndarray): Image data
    """
    print("\n=== Image Analysis ===")
    print(f"Shape: {img_array.shape}")
    print(f"Data type: {img_array.dtype}")
    print(f"Min pixel value: {img_array.min()}")
    print(f"Max pixel value: {img_array.max()}")
    print(f"Mean pixel value: {img_array.mean():.2f}")
    print(f"Standard deviation: {img_array.std():.2f}")
    print(f"Unique values count: {len(np.unique(img_array))}")
    
    # Check if it's actually using 10-bit range
    if img_array.max() <= 1023:
        print("Image appears to be using 10-bit range (0-1023)")
    else:
        print("Image may not be using 10-bit range")

def main():
    """Main function to demonstrate reading 10-bit TIFF files."""
    
    # Path to the dataset
    dataset_path = Path("IPS_Dataset")
    
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        return
    
    # Get list of TIFF files
    tiff_files = list(dataset_path.glob("*.tiff"))
    
    if not tiff_files:
        print("No TIFF files found in the dataset directory")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Read the first image (or a specific one)
    first_image = tiff_files[0]
    print(f"\nReading: {first_image.name}")
    
    # Read the 10-bit GeoTIFF
    img_array = read_10bit_geotiff(first_image)
    
    if img_array is not None:
        # Analyze the image
        analyze_image_properties(img_array)
        
        # Display the image
        display_image(img_array, f"10-bit TIFF: {first_image.name}")
        
        # Save as 10-bit TIFF preserving the original precision
        output_path = f"10bit_{first_image.stem}.tiff"
        save_as_10bit_tiff(img_array, output_path)
        
        # Example: Process multiple images
        print(f"\nProcessing first 5 images...")
        for i, tiff_file in enumerate(tiff_files[:5]):
            print(f"\n{i+1}. Processing {tiff_file.name}")
            img = read_10bit_geotiff(tiff_file)
            if img is not None:
                print(f"   Shape: {img.shape}, Range: {img.min()}-{img.max()}")

if __name__ == "__main__":
    main()
