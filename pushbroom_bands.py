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
from scipy import signal
from scipy.ndimage import shift
from scipy.signal import savgol_filter
from skimage.registration import phase_cross_correlation

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

def compute_correlation_confidence(correlation_surface):
    """
    Compute confidence metrics for correlation peak.
    
    Args:
        correlation_surface (numpy.ndarray): Correlation values
        
    Returns:
        dict: Confidence metrics including peak-to-mean ratio and sharpness
    """
    if len(correlation_surface) < 3:
        return {'peak_to_mean': 0.0, 'sharpness': 0.0, 'confidence': 0.0}
    
    peak_val = np.max(correlation_surface)
    mean_val = np.mean(correlation_surface)
    
    # Peak-to-mean ratio
    peak_to_mean = peak_val / (mean_val + 1e-8)
    
    # Find second highest peak for sharpness measure
    peak_idx = np.argmax(correlation_surface)
    temp_corr = correlation_surface.copy()
    temp_corr[max(0, peak_idx-1):min(len(temp_corr), peak_idx+2)] = -np.inf
    second_peak = np.max(temp_corr) if len(temp_corr[temp_corr > -np.inf]) > 0 else mean_val
    
    # Sharpness as peak-to-second-peak ratio
    sharpness = peak_val / (second_peak + 1e-8)
    
    # Combined confidence score
    confidence = min(peak_to_mean / 3.0, 1.0) * min(sharpness / 2.0, 1.0)
    
    return {
        'peak_to_mean': peak_to_mean,
        'sharpness': sharpness, 
        'confidence': confidence
    }

def sub_pixel_peak_refinement(correlation_surface, peak_idx):
    """
    Refine peak location to sub-pixel accuracy using parabolic fitting.
    
    Args:
        correlation_surface (numpy.ndarray): Correlation values
        peak_idx (int): Integer peak location
        
    Returns:
        float: Sub-pixel refined peak location
    """
    if peak_idx <= 0 or peak_idx >= len(correlation_surface) - 1:
        return float(peak_idx)
    
    # Get three points around peak
    y1 = correlation_surface[peak_idx - 1]
    y2 = correlation_surface[peak_idx]
    y3 = correlation_surface[peak_idx + 1]
    
    # Parabolic interpolation
    denom = 2 * (y1 - 2*y2 + y3)
    if abs(denom) < 1e-8:
        return float(peak_idx)
    
    sub_pixel_offset = (y1 - y3) / denom
    return peak_idx + sub_pixel_offset

def select_best_band_for_correlation(bands_dict, correlation_history=None):
    """
    Select the best band for correlation based on contrast and historical performance.
    
    Args:
        bands_dict (dict): Dictionary of band data
        correlation_history (dict): Historical confidence scores per band
        
    Returns:
        str: Best band name for correlation
    """
    if correlation_history is None:
        correlation_history = {}
    
    band_scores = {}
    preferred_order = ['nir', 'green_pan', 'red', 'blue', 'red_edge']
    
    for band_name in preferred_order:
        if band_name in bands_dict:
            band_data = bands_dict[band_name]
            
            # Compute contrast (standard deviation)
            contrast = np.std(band_data.astype(np.float32))
            
            # Historical performance weight
            hist_weight = correlation_history.get(band_name, 0.5)
            
            # Combined score
            band_scores[band_name] = contrast * (0.7 + 0.3 * hist_weight)
    
    if not band_scores:
        return list(bands_dict.keys())[0] if bands_dict else None
    
    return max(band_scores, key=band_scores.get)

def detect_drift_enhanced_correlation(previous_band, current_band, max_shift=50, 
                                    enable_subpixel=True, adaptive_overlap=True):
    """
    Enhanced drift detection using cross-correlation with sub-pixel accuracy,
    confidence metrics, and adaptive overlap sizing.
    
    Args:
        previous_band (numpy.ndarray): Previous band image
        current_band (numpy.ndarray): Current band image to align
        max_shift (int): Maximum expected shift in y-direction (pixels)
        enable_subpixel (bool): Enable sub-pixel peak refinement
        adaptive_overlap (bool): Use adaptive overlap sizing
        
    Returns:
        dict: Drift information including shift, confidence, and metrics
    """
    # Get dimensions
    h_prev, w_prev = previous_band.shape
    h_curr, w_curr = current_band.shape
    
    # Adaptive overlap sizing
    if adaptive_overlap:
        min_height = min(h_prev, h_curr)
        # Base overlap on expected drift + safety factor
        overlap_size = min(min_height // 3, max_shift * 3)
        overlap_size = max(overlap_size, 15)  # Minimum overlap
    else:
        min_height = min(h_prev, h_curr)
        overlap_size = min(min_height // 2, max_shift * 2)
        overlap_size = max(overlap_size, 10)
    
    # Extract overlapping regions
    prev_region = previous_band[-overlap_size:, :]  # Last portion of previous
    curr_region = current_band[:overlap_size, :]    # First portion of current
    
    # Convert to float for better correlation
    prev_float = prev_region.astype(np.float32)
    curr_float = curr_region.astype(np.float32)
    
    # Enhanced normalization with contrast stretching
    def normalize_region(region):
        # Remove low-frequency components to emphasize edges
        mean_val = np.mean(region)
        norm_region = region - mean_val
        std_val = np.std(norm_region)
        if std_val > 0:
            norm_region = norm_region / std_val
        return norm_region
    
    prev_norm = normalize_region(prev_float)
    curr_norm = normalize_region(curr_float)
    
    # Use scikit-image phase correlation for better sub-pixel accuracy
    try:
        shift_yx, error, phase_diff = phase_cross_correlation(
            prev_norm, curr_norm, upsample_factor=10
        )
        drift_y = shift_yx[0]  # y-direction shift
        confidence_score = 1.0 - error  # Higher confidence = lower error
        
        # Limit drift to reasonable range
        drift_y = np.clip(drift_y, -max_shift, max_shift)
        
        return {
            'drift_y': drift_y,
            'confidence': max(confidence_score, 0.0),
            'method': 'phase_correlation',
            'overlap_size': overlap_size,
            'error': error
        }
        
    except Exception as e:
        # Fallback to original method with enhancements
        print(f"    Phase correlation failed, using fallback: {e}")
        
        # Compute cross-correlation column-wise
        correlations = []
        valid_columns = 0
        
        for col in range(prev_norm.shape[1]):
            if np.std(prev_norm[:, col]) > 0.1 and np.std(curr_norm[:, col]) > 0.1:
                corr = np.correlate(prev_norm[:, col], curr_norm[:, col], mode='full')
                correlations.append(corr)
                valid_columns += 1
        
        if not correlations:
            return {
                'drift_y': 0.0,
                'confidence': 0.0,
                'method': 'fallback_failed',
                'overlap_size': overlap_size,
                'error': 1.0
            }
        
        # Average correlation across columns
        avg_correlation = np.mean(correlations, axis=0)
        
        # Find integer peak
        peak_idx = np.argmax(avg_correlation)
        
        # Sub-pixel refinement
        if enable_subpixel:
            refined_peak = sub_pixel_peak_refinement(avg_correlation, peak_idx)
        else:
            refined_peak = float(peak_idx)
        
        # Calculate drift
        drift_y = refined_peak - (overlap_size - 1)
        
        # Compute confidence
        confidence_metrics = compute_correlation_confidence(avg_correlation)
        
        # Adjust confidence based on number of valid columns
        column_weight = min(valid_columns / (prev_norm.shape[1] * 0.5), 1.0)
        final_confidence = confidence_metrics['confidence'] * column_weight
        
        # Limit drift to reasonable range
        drift_y = np.clip(drift_y, -max_shift, max_shift)
        
        return {
            'drift_y': drift_y,
            'confidence': final_confidence,
            'method': 'enhanced_correlation',
            'overlap_size': overlap_size,
            'metrics': confidence_metrics,
            'valid_columns': valid_columns
        }

class DriftSmoother:
    """
    Temporal smoothing for drift corrections using Savitzky-Golay filter.
    """
    def __init__(self, window_length=7, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
        self.drift_history = []
        self.confidence_history = []
        
    def add_measurement(self, drift_y, confidence):
        """Add new drift measurement."""
        self.drift_history.append(drift_y)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        max_history = self.window_length * 2
        if len(self.drift_history) > max_history:
            self.drift_history = self.drift_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]
    
    def get_smoothed_drift(self, current_drift, current_confidence, confidence_threshold=0.3):
        """
        Get smoothed drift value considering confidence.
        
        Args:
            current_drift (float): Current frame drift
            current_confidence (float): Confidence in current measurement
            confidence_threshold (float): Minimum confidence to use measurement
            
        Returns:
            float: Smoothed drift value
        """
        if current_confidence < confidence_threshold and self.drift_history:
            # Low confidence - use weighted average of recent history
            weights = np.array(self.confidence_history[-5:])
            drifts = np.array(self.drift_history[-5:])
            if np.sum(weights) > 0:
                return np.average(drifts, weights=weights)
            else:
                return self.drift_history[-1]  # Last known value
        
        # Add to history
        self.add_measurement(current_drift, current_confidence)
        
        # Apply smoothing if we have enough data
        if len(self.drift_history) >= self.window_length:
            try:
                smoothed_values = savgol_filter(
                    self.drift_history, 
                    self.window_length, 
                    self.polyorder
                )
                return smoothed_values[-1]
            except:
                return current_drift
        
        return current_drift

def correct_drift_y_shift(image, drift_y):
    """
    Correct drift in y-direction by applying the detected shift with sub-pixel accuracy.
    
    Args:
        image (numpy.ndarray): Image to correct
        drift_y (float): Vertical drift in pixels (can be sub-pixel)
        
    Returns:
        numpy.ndarray: Drift-corrected image
    """
    if abs(drift_y) < 1e-6:
        return image
    
    # Use scipy's shift function for sub-pixel accuracy
    # Only shift in y-direction (axis=0)
    corrected = shift(image, shift=(drift_y, 0), mode='constant', cval=0)
    
    return corrected

def create_pushbroom_image_enhanced(band_data_list, band_name, fps_pixels=25, 
                                   enable_drift_correction=True, max_shift=50,
                                   enable_subpixel=True, enable_smoothing=True,
                                   confidence_threshold=0.3):
    """
    Create a pushbroom image with enhanced drift correction featuring:
    - Sub-pixel accuracy drift detection
    - Confidence-based measurement filtering
    - Temporal smoothing of drift corrections
    - Adaptive overlap sizing
    
    Args:
        band_data_list (list): List of band data arrays
        band_name (str): Name of the band
        fps_pixels (int): Number of pixels to use from each subsequent image
        enable_drift_correction (bool): Enable drift correction
        max_shift (int): Maximum expected shift in y-direction
        enable_subpixel (bool): Enable sub-pixel accuracy
        enable_smoothing (bool): Enable temporal smoothing
        confidence_threshold (float): Minimum confidence for drift measurements
        
    Returns:
        numpy.ndarray: Stitched pushbroom image
    """
    if not band_data_list:
        return None
    
    # Start with the first image (all pixels)
    pushbroom = band_data_list[0].copy()
    reference_band = band_data_list[0].copy()
    
    # Initialize drift smoother if enabled
    drift_smoother = DriftSmoother() if enable_smoothing else None
    
    drift_corrections = []
    confidence_scores = []
    low_confidence_count = 0
    
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
        
        # Apply enhanced drift correction if enabled
        if enable_drift_correction:
            try:
                # Enhanced drift detection
                drift_result = detect_drift_enhanced_correlation(
                    reference_band, selected_pixels, max_shift, 
                    enable_subpixel, adaptive_overlap=True
                )
                
                raw_drift_y = drift_result['drift_y']
                confidence = drift_result['confidence']
                method = drift_result['method']
                
                # Apply temporal smoothing if enabled
                if enable_smoothing and drift_smoother:
                    smoothed_drift_y = drift_smoother.get_smoothed_drift(
                        raw_drift_y, confidence, confidence_threshold
                    )
                else:
                    smoothed_drift_y = raw_drift_y
                
                # Apply drift correction
                corrected_pixels = correct_drift_y_shift(selected_pixels, smoothed_drift_y)
                
                # Store drift correction info
                drift_corrections.append(smoothed_drift_y)
                confidence_scores.append(confidence)
                
                if confidence < confidence_threshold:
                    low_confidence_count += 1
                
                # Detailed logging
                if confidence < confidence_threshold:
                    print(f"    Frame {i}: Low confidence drift (dy={smoothed_drift_y:.2f}, "
                          f"conf={confidence:.2f}, method={method}) - corrected with smoothing")
                else:
                    print(f"    Frame {i}: Drift detected (dy={smoothed_drift_y:.2f}, "
                          f"conf={confidence:.2f}, method={method}) - corrected")
                
                # Update reference for next iteration (use corrected image)
                reference_band = corrected_pixels
                selected_pixels = corrected_pixels
                
            except Exception as e:
                print(f"    Frame {i}: Enhanced drift correction failed - {e}")
                # Fallback to no correction
                drift_corrections.append(0.0)
                confidence_scores.append(0.0)
                low_confidence_count += 1
        else:
            print(f"    Frame {i}: No drift correction applied")
        
        # Append to the end of the pushbroom
        pushbroom = np.vstack([selected_pixels, pushbroom])
    
    # Print enhanced drift correction summary
    if enable_drift_correction and drift_corrections:
        avg_dy = np.mean(drift_corrections)
        max_dy = max([abs(d) for d in drift_corrections])
        std_dy = np.std(drift_corrections)
        avg_conf = np.mean(confidence_scores) if confidence_scores else 0.0
        
        print(f"  Enhanced drift correction summary:")
        print(f"    Avg drift: {avg_dy:.2f}px, Max: {max_dy:.2f}px, Std: {std_dy:.2f}px")
        print(f"    Avg confidence: {avg_conf:.2f}, Low confidence frames: {low_confidence_count}")
        
        if enable_subpixel:
            print(f"    Sub-pixel accuracy enabled")
        if enable_smoothing:
            print(f"    Temporal smoothing enabled")
    
    print(f"  {band_name} pushbroom: shape {pushbroom.shape} (first image: all pixels, others: {fps_pixels} pixels from start)")
    
    return pushbroom

def create_pushbroom_image(band_data_list, band_name, fps_pixels=25, enable_drift_correction=True, max_shift=50):
    """
    Legacy function - calls enhanced version with default settings for backward compatibility.
    """
    return create_pushbroom_image_enhanced(
        band_data_list, band_name, fps_pixels, enable_drift_correction, max_shift,
        enable_subpixel=True, enable_smoothing=True, confidence_threshold=0.3
    )

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

def create_rgb_image(band_data_lists, fps_pixels=25, enable_drift_correction=True, max_shift=50):
    """
    Create RGB pushbroom images from the spectral bands with enhanced drift correction.
    Uses Red, Green-pan, and Blue bands for RGB composition.
    
    Args:
        band_data_lists (dict): Dictionary containing band data for each band
        fps_pixels (int): Number of pixels per second
        enable_drift_correction (bool): Enable enhanced drift correction
        max_shift (int): Maximum expected shift in y-direction
        
    Returns:
        dict: Dictionary containing RGB pushbroom images
    """
    rgb_images = {}
    
    # Check if we have the required bands
    required_bands = ['red', 'green_pan', 'blue']
    if not all(band in band_data_lists for band in required_bands):
        print("Warning: Missing required bands for RGB creation")
        return rgb_images
    
    print("\nCreating RGB pushbroom images with enhanced drift correction...")
    
    # Create RGB pushbroom for each band combination
    for band_name in required_bands:
        if band_data_lists[band_name]:
            print(f"\nCreating RGB pushbroom for {band_name} band...")
            
            # Create pushbroom image for this band with enhanced drift correction
            pushbroom = create_pushbroom_image_enhanced(
                band_data_lists[band_name], f"rgb_{band_name}", 
                fps_pixels, enable_drift_correction, max_shift,
                enable_subpixel=True, enable_smoothing=True, confidence_threshold=0.3
            )
            
            if pushbroom is not None:
                rgb_images[band_name] = pushbroom
                print(f"  RGB {band_name} pushbroom: shape {pushbroom.shape}")
    
    # Create composite RGB image using all three bands
    if len(rgb_images) == 3:
        print(f"\nCreating composite RGB pushbroom...")
        
        # Get the minimum height to ensure all bands have the same dimensions
        min_height = min(img.shape[0] for img in rgb_images.values())
        
        # Resize all bands to the same height
        resized_bands = {}
        for band_name, img in rgb_images.items():
            if img.shape[0] > min_height:
                # Take the first min_height rows
                resized_bands[band_name] = img[:min_height, :]
            else:
                resized_bands[band_name] = img
        
        # Create RGB composite (Red, Green, Blue)
        rgb_composite = np.stack([
            resized_bands['red'],
            resized_bands['green_pan'], 
            resized_bands['blue']
        ], axis=2)
        
        print(f"  RGB composite shape: {rgb_composite.shape}")
        rgb_images['composite'] = rgb_composite
    
    return rgb_images

def save_rgb_image(rgb_array, output_path):
    """
    Save RGB image as 8-bit PNG for display.
    
    Args:
        rgb_array (numpy.ndarray): RGB image data (0-1023 range)
        output_path (str): Output file path
    """
    # Scale from 10-bit (0-1023) to 8-bit (0-255)
    rgb_8bit = (rgb_array / 1023.0 * 255).astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(rgb_8bit).save(output_path)
    print(f"  Saved RGB: {output_path}")

def main():
    """Main function to process images and create pushbroom bands."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create pushbroom images from 10-bit GeoTIFF files')
    parser.add_argument('--fps-pixels', type=int, default=25, 
                       help='Number of pixels per second (default: 25)')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to process (default: 10)')
    parser.add_argument('--no-drift-correction', action='store_true',
                       help='Disable drift correction (default: enabled)')
    parser.add_argument('--max-shift', type=int, default=50,
                       help='Maximum expected shift in y-direction for drift correction (default: 50)')
    
    args = parser.parse_args()
    
    # Configuration variables
    fps_pixels = args.fps_pixels
    num_images = args.num_images
    enable_drift_correction = not args.no_drift_correction
    max_shift = args.max_shift
    
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
    print(f"Drift correction: {'Enabled' if enable_drift_correction else 'Disabled'}")
    if enable_drift_correction:
        print(f"Maximum shift: {max_shift} pixels")
    
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
            
            # Create pushbroom image with fps_pixels parameter and drift correction
            pushbroom = create_pushbroom_image(band_data_list, band_name, fps_pixels, enable_drift_correction, max_shift)
            
            if pushbroom is not None:
                # Save pushbroom image
                output_path = f"pushbroom_{band_name}_{num_images}images_{fps_pixels}fps.tiff"
                save_10bit_tiff(pushbroom, output_path)
                
                # Also save individual band from first image for reference
                if band_data_list:
                    single_band_path = f"single_{band_name}_band.tiff"
                    save_10bit_tiff(band_data_list[0], single_band_path)
        else:
            print(f"No data available for {band_name} band")
    
    # Create RGB pushbroom images
    rgb_images = create_rgb_image(band_data_lists, fps_pixels, enable_drift_correction, max_shift)
    
    # Save RGB images
    if rgb_images:
        print(f"\nSaving RGB pushbroom images...")
        
        for rgb_name, rgb_array in rgb_images.items():
            if rgb_name == 'composite':
                # Save composite RGB as PNG for display
                output_path = f"pushbroom_rgb_composite_{num_images}images_{fps_pixels}fps.png"
                save_rgb_image(rgb_array, output_path)
            else:
                # Save individual RGB bands as TIFF
                output_path = f"pushbroom_rgb_{rgb_name}_{num_images}images_{fps_pixels}fps.tiff"
                save_10bit_tiff(rgb_array, output_path)
    
    print(f"\nPushbroom processing complete!")
    print(f"Created pushbroom images for {num_images} images using {fps_pixels} pixels per second")
    if rgb_images:
        print(f"Created RGB pushbroom images including composite RGB")

if __name__ == "__main__":
    main()
