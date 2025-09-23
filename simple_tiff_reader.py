#!/usr/bin/env python3
"""
Simple script to read a single 10-bit GeoTIFF file.
Handles 10-bit data that may be scaled by 16 in the GeoTIFF format.
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage

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

def extract_spectral_bands(image_data):
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

def calculate_autocorrelation(img1, img2, x_range=(-3, 3), y_range=(27, 37)):
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
        tuple: (best_x_shift, best_y_shift, max_correlation, correlation_map)
    """
    print(f"\nCalculating autocorrelation...")
    print(f"X shift range: {x_range[0]} to {x_range[1]} pixels (+ = img2 shifts right)")
    print(f"Y shift range: {y_range[0]} to {y_range[1]} pixels (+ = img2 shifts up)")
    
    # Convert to float for calculations
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    # Normalize images to have zero mean and unit variance
    img1_norm = (img1_float - np.mean(img1_float)) / np.std(img1_float)
    img2_norm = (img2_float - np.mean(img2_float)) / np.std(img2_float)
    
    # Initialize correlation map
    x_shifts = range(x_range[0], x_range[1] + 1)
    y_shifts = range(y_range[0], y_range[1] + 1)
    
    correlation_map = np.zeros((len(y_shifts), len(x_shifts)))
    
    best_correlation = -np.inf
    best_x_shift = 0
    best_y_shift = 0
    
    # Test each shift combination
    for i, y_shift in enumerate(y_shifts):
        for j, x_shift in enumerate(x_shifts):
            print(f"    Testing shift: x={x_shift}, y={y_shift}")
            
            # Calculate overlapping regions when img2 is shifted by (x_shift, y_shift)
            # Positive y_shift: img2 shifted UP relative to img1
            # Positive x_shift: img2 shifted RIGHT relative to img1
            
            # Calculate valid overlap bounds for UP shift (positive y_shift)
            # If img2 is shifted UP by y_shift, then:
            # - img1 region starts from 0, img2 region starts from y_shift
            y_overlap_start = max(0, -y_shift)  # Start y in img1 coordinate system
            y_overlap_end = min(img1.shape[0], img2.shape[0] - y_shift)  # End y in img1 coordinate system
            x_overlap_start = max(0, x_shift)  # Start x in img1 coordinate system
            x_overlap_end = min(img1.shape[1], img2.shape[1] + x_shift)  # End x in img1 coordinate system
            
            # Extract regions from img1 (reference frame)
            y1_start = y_overlap_start
            y1_end = y_overlap_end
            x1_start = x_overlap_start
            x1_end = x_overlap_end
            
            # Extract regions from img2 (shifted frame, UP by y_shift)
            y2_start = y_overlap_start + y_shift  # Convert to img2 coordinates
            y2_end = y_overlap_end + y_shift
            x2_start = x_overlap_start - x_shift  # Convert to img2 coordinates
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
                    # Use a more efficient correlation calculation
                    mean1 = np.mean(region1)
                    mean2 = np.mean(region2)
                    std1 = np.std(region1)
                    std2 = np.std(region2)
                    
                    if std1 > 0 and std2 > 0:
                        correlation = np.mean((region1 - mean1) * (region2 - mean2)) / (std1 * std2)
                    else:
                        correlation = 0
                    
                    correlation_map[i, j] = correlation
                    print(f"      Correlation: {correlation:.4f}")
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_x_shift = x_shift
                        best_y_shift = y_shift
                        print(f"      New best: {correlation:.4f} at ({x_shift}, {y_shift})")
    
    print(f"Best match found:")
    print(f"  X shift: {best_x_shift} pixels (img2 relative to img1)")
    print(f"  Y shift: {best_y_shift} pixels (img2 relative to img1)")
    print(f"  Correlation: {best_correlation:.4f}")
    
    return best_x_shift, best_y_shift, best_correlation, correlation_map

def align_images(img1, img2, x_shift, y_shift):
    """
    Align two images based on the calculated shift values.
    Uses the same logic as the correlation calculation.
    
    Args:
        img1 (numpy.ndarray): Reference image
        img2 (numpy.ndarray): Image to shift
        x_shift (int): X-axis shift in pixels (positive = img2 shifted right)
        y_shift (int): Y-axis shift in pixels (positive = img2 shifted up)
        
    Returns:
        tuple: (aligned_img1_region, aligned_img2_region)
    """
    # Use the same logic as the correlation calculation
    # Calculate valid overlap bounds for UP shift (positive y_shift)
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
    
    # Extract aligned regions
    aligned_img1 = img1[y1_start:y1_end, x1_start:x1_end]
    aligned_img2 = img2[y2_start:y2_end, x2_start:x2_end]
    
    # Ensure both regions have the same size
    min_h = min(aligned_img1.shape[0], aligned_img2.shape[0])
    min_w = min(aligned_img1.shape[1], aligned_img2.shape[1])
    
    aligned_img1 = aligned_img1[:min_h, :min_w]
    aligned_img2 = aligned_img2[:min_h, :min_w]
    
    return aligned_img1, aligned_img2

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
    file_path = "IPS_Dataset/image_003712.tiff"
    file_path2 = "IPS_Dataset/image_003713.tiff"  # Alternative extension
    
    try:
        # Read the 10-bit GeoTIFF
        image_data = read_geotiff_10bit(file_path)
        print(f"\nSuccessfully loaded image with shape {image_data.shape}")
        print(f"Final 10-bit range: {image_data.min()} - {image_data.max()}")
        
        # Save as 10-bit TIFF preserving the original precision
        save_as_10bit_tiff(image_data)

        # Optional: load the second image to compare
        image_data2 = read_geotiff_10bit(file_path2)
        print(f"\nSuccessfully loaded second image with shape {image_data2.shape}")
        print(f"Final 10-bit range: {image_data2.min()} - {image_data2.max()}")

        # Save the second image as well
        save_as_10bit_tiff(image_data2, output_path="converted_10bit_2.tiff")

        # Extract spectral bands from both images
        print(f"\nExtracting spectral bands from image 1:")
        bands1 = extract_spectral_bands(image_data)
        print(f"\nExtracting spectral bands from image 2:")
        bands2 = extract_spectral_bands(image_data2)

        # Calculate autocorrelation using green_pan band only
        print(f"\nPerforming autocorrelation on green_pan band...")
        green_pan1 = bands1['green_pan']
        green_pan2 = bands2['green_pan']
        
        best_x, best_y, max_corr, corr_map = calculate_autocorrelation(
            green_pan1, green_pan2, 
            x_range=(-5, 5), 
            y_range=(27, 37)
        )

        # Get aligned image regions (using green_pan bands)
        aligned_green1, aligned_green2 = align_images(green_pan1, green_pan2, best_x, best_y)
        print(f"\nAligned green_pan regions shape: {aligned_green1.shape}")

        # Combined correlation map and difference plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Correlation map
        ax1.set_title(f"Correlation Map\nBest Match: X={best_x}, Y={best_y}, Correlation={max_corr:.4f}")
        im1 = ax1.imshow(corr_map, cmap='viridis', aspect='auto')
        ax1.set_xlabel('X Shift (pixels)')
        ax1.set_ylabel('Y Shift (pixels)')

        # Set axis labels to show actual shift values
        x_ticks = range(len(range(-5, 6)))
        y_ticks = range(len(range(27, 38)))
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(range(-5, 6))
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(range(27, 38))
        
        plt.colorbar(im1, ax=ax1, label='Correlation')
        
        # Mark the best correlation point
        best_x_idx = best_x - (-5)  # Convert to index
        best_y_idx = best_y - 27
        ax1.plot(best_x_idx, best_y_idx, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2, label=f'Best Match ({best_x}, {best_y})')
        ax1.legend()
        
        # Difference plot
        ax2.set_title("Difference between Aligned Green_pan Bands")
        diff = aligned_green1.astype(np.float32) - aligned_green2.astype(np.float32)
        im2 = ax2.imshow(diff, cmap='RdBu', vmin=-20, vmax=20)
        plt.colorbar(im2, ax=ax2, label='Pixel Difference', orientation='horizontal')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        plt.show()

        # Display green_pan bands and aligned regions
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
        
        # Green_pan Band 1
        axs[0, 0].set_title("Green_pan Band 1 (Reference)")
        axs[0, 0].imshow(green_pan1, cmap='gray', vmin=200, vmax=1023)
        
        # Green_pan Band 2
        axs[0, 1].set_title("Green_pan Band 2")
        axs[0, 1].imshow(green_pan2, cmap='gray', vmin=200, vmax=1023)
        
        # Aligned Green_pan 1
        axs[1, 0].set_title(f"Aligned Green_pan 1\nShape: {aligned_green1.shape}")
        axs[1, 0].imshow(aligned_green1, cmap='gray', vmin=200, vmax=1023)
        
        # Aligned Green_pan 2
        axs[1, 1].set_title(f"Aligned Green_pan 2\nShape: {aligned_green2.shape}")
        axs[1, 1].imshow(aligned_green2, cmap='gray', vmin=200, vmax=1023)
        
        plt.tight_layout()
        plt.show()

        

    except Exception as e:
        print(f"Error: {e}")