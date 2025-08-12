import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy import ndimage
import csv
import math

# --- Configuration Parameters ---
IMAGE_PATH = 'image.png'

# Scale bar properties
SCALE_BAR_MM = 2.0
SCALE_BAR_ROI_COORDS = (315, 330, 945, 1150)

# --- Multiple Purple Detection Strategies ---

def method1_expanded_hsv_range(image):
    """Method 1: Expand HSV range to catch more purple variations"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # More permissive range - catch edge cases
    lower_purple = np.array([120, 80, 5])    # Lower hue, lower saturation, lower value
    upper_purple = np.array([170, 255, 255]) # Higher hue range
    
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
    
    # Gentler morphological operations
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask, "Expanded HSV Range"


def method2_multiple_hsv_ranges(image):
    """Method 2: Use multiple HSV ranges to catch different purple shades"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define multiple purple ranges
    purple_ranges = [
        ([120, 100, 10], [140, 255, 255]),  # Darker purple
        ([140, 100, 10], [160, 255, 255]),  # Medium purple  
        ([160, 100, 10], [170, 255, 255]),  # Lighter purple
        ([125, 80, 5], [165, 255, 255]),    # Your original range but wider
    ]
    
    # Combine all masks
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    for lower, upper in purple_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv_image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Clean up the combined mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return combined_mask, "Multiple HSV Ranges"


def method3_saturation_based(image):
    """Method 3: Focus on high saturation regions (your HSV shows this works well)"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract saturation channel
    saturation = hsv_image[:, :, 1]
    
    # Threshold based on saturation (your image shows bright spots = high saturation)
    _, sat_mask = cv2.threshold(saturation, 100, 255, cv2.THRESH_BINARY)
    
    # Also consider hue to avoid other high-saturation colors
    hue = hsv_image[:, :, 0]
    hue_mask = cv2.inRange(hue, 120, 170)
    
    # Combine saturation and hue constraints
    combined_mask = cv2.bitwise_and(sat_mask, hue_mask)
    
    return combined_mask, "Saturation-Based Detection"


def method4_adaptive_threshold(image):
    """Method 4: Use adaptive thresholding on saturation channel"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1]
    
    # Adaptive threshold on saturation channel
    adaptive_mask = cv2.adaptiveThreshold(
        saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, -10
    )
    
    # Filter by hue
    hue_mask = cv2.inRange(hsv_image[:, :, 0], 120, 170)
    combined_mask = cv2.bitwise_and(adaptive_mask, hue_mask)

     # 🔍 Dilate to enlarge the dot regions (makes contours wrap entire dot)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    return combined_mask, "Adaptive Threshold"


def method5_statistical_outlier_detection(image):
    """Method 5: Statistical approach - find pixels that deviate significantly"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate statistics for each channel
    h_mean, h_std = np.mean(hsv_image[:, :, 0]), np.std(hsv_image[:, :, 0])
    s_mean, s_std = np.mean(hsv_image[:, :, 1]), np.std(hsv_image[:, :, 1])
    
    # Find pixels with high saturation (outliers)
    high_sat_mask = hsv_image[:, :, 1] > (s_mean + 2 * s_std)
    
    # Filter by purple hue range
    purple_hue_mask = cv2.inRange(hsv_image[:, :, 0], 120, 170)
    
    # Combine conditions
    combined_mask = np.logical_and(high_sat_mask, purple_hue_mask).astype(np.uint8) * 255
    
    return combined_mask, "Statistical Outlier Detection"


def method6_lab_color_space(image):
    """Method 6: Use LAB color space for better purple detection"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # In LAB space, purple typically has:
    # L: variable lightness
    # A: positive (magenta/red direction) 
    # B: negative (blue direction)
    
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]
    
    # Purple detection in LAB
    # A > 128 (towards magenta), B < 128 (towards blue)
    purple_mask = np.logical_and(a_channel > 135, b_channel < 120).astype(np.uint8) * 255
    
    return purple_mask, "LAB Color Space"


def detect_contours_with_flexible_filtering(mask, image, method_name):
    """Detect contours with more flexible size filtering"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], []
    
    # Calculate area statistics
    areas = [cv2.contourArea(cnt) for cnt in contours]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    
    # More flexible filtering - keep anything within reasonable range
    min_area = max(3, mean_area - 2 * std_area)  # Very small minimum
    max_area = max(5000, mean_area + 6 * std_area)

    dot_centroids = []
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dot_centroids.append((cx, cy))
                valid_contours.append(contour)
    
    print(f"{method_name}: Detected {len(dot_centroids)} dots (area range: {min_area:.1f} - {max_area:.1f})")
    return dot_centroids, valid_contours


def compare_detection_methods(image):
    """Compare all detection methods side by side"""
    methods = [
        method1_expanded_hsv_range,
        method2_multiple_hsv_ranges,
        method3_saturation_based,
        method4_adaptive_threshold,
        method5_statistical_outlier_detection,
        method6_lab_color_space
    ]
    
    results = {}
    
    # Create subplot for comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        mask, method_name = method(image)
        centroids, contours = detect_contours_with_flexible_filtering(mask, image, method_name)
        
        results[method_name] = {
            'mask': mask,
            'centroids': centroids,
            'contours': contours,
            'count': len(centroids)
        }
        
        # Visualize results
        result_image = image.copy()
        
        # Draw contours
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 1)
        
        # Draw centroids
        for cx, cy in centroids:
            cv2.circle(result_image, (cx, cy), 3, (0, 0, 255), -1)
        
        # Convert to RGB for matplotlib
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(result_rgb)
        axes[i].set_title(f'{method_name}\n{len(centroids)} dots detected')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n=== DETECTION COMPARISON SUMMARY ===")
    for method_name, result in results.items():
        print(f"{method_name:30s}: {result['count']:4d} dots")
    
    return results


def create_optimized_detection_function(image):
    """Create an optimized detection function based on your specific image"""
    
    # Based on your HSV image, method 3 (saturation-based) looks most promising
    # Let's create an optimized version
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Your HSV image shows that purple dots have very high saturation
    # Use multiple approaches and combine them
    
    # Approach 1: High saturation + purple hue
    sat_mask = cv2.threshold(hsv_image[:, :, 1], 120, 255, cv2.THRESH_BINARY)[1]
    hue_mask = cv2.inRange(hsv_image[:, :, 0], 125, 165)
    mask1 = cv2.bitwise_and(sat_mask, hue_mask)
    
    # Approach 2: Your original range but with lower thresholds
    lower_purple = np.array([120, 90, 5])
    upper_purple = np.array([170, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_purple, upper_purple)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask1, mask2)
    
    # Gentle cleanup
    kernel = np.ones((2, 2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask


def calibrate_scale(image, roi_coords, known_length_mm):
    """Your existing calibration function"""
    y_start, y_end, x_start, x_end = roi_coords
    if not (0 <= y_start < y_end <= image.shape[0] and 
            0 <= x_start < x_end <= image.shape[1]):
        print(f"Error: Scale bar ROI coordinates {roi_coords} are out of image bounds {image.shape[:2]}.")
        return None
    
    roi = image[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        print("Error: Scale bar ROI is empty. Check coordinates.")
        return None
        
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found in scale bar ROI.")
        return None
        
    scale_bar_contour = max(contours, key=cv2.contourArea)
    x_bar, y_bar, w_bar, h_bar = cv2.boundingRect(scale_bar_contour)
    scale_bar_pixel_length = w_bar
    
    if scale_bar_pixel_length == 0:
        print("Error: Detected scale bar has zero pixel length.")
        return None
        
    pixels_per_mm = scale_bar_pixel_length / known_length_mm
    print(f"Scale bar detected: {scale_bar_pixel_length} pixels = {known_length_mm} mm")
    print(f"Calibration: {pixels_per_mm:.2f} pixels/mm")
    return pixels_per_mm


def main():
    """Enhanced main function with method comparison"""
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels.")
    
    # Scale calibration
    pixels_per_mm = calibrate_scale(image, SCALE_BAR_ROI_COORDS, SCALE_BAR_MM)
    if pixels_per_mm is None:
        print("Exiting due to scale calibration failure.")
        return

    print("\n=== COMPARING DETECTION METHODS ===")
    results = compare_detection_methods(image)
    
    print("\n=== USING OPTIMIZED DETECTION ===")
    optimized_mask = create_optimized_detection_function(image)
    optimized_centroids, optimized_contours = detect_contours_with_flexible_filtering(
        optimized_mask, image, "Optimized Combined Method"
    )

        # --- DISTANCE ANALYSIS BETWEEN CENTROIDS ---

    if len(optimized_centroids) >= 2:
        # Build KDTree for efficient distance computation
        tree = KDTree(optimized_centroids)
        distances = []

        for i, point in enumerate(optimized_centroids):
            # Query for the nearest neighbor (excluding self)
            dists, idxs = tree.query(point, k=2)
            nearest_dist = dists[1]  # dists[0] is 0 (self)
            distances.append(nearest_dist)

        # Convert pixel distances to mm if calibration succeeded
        if pixels_per_mm:
            distances_mm = [d / pixels_per_mm for d in distances]
            avg_distance_mm = sum(distances_mm) / len(distances_mm)
            unit = "mm"
        else:
            distances_mm = distances
            avg_distance_mm = sum(distances) / len(distances)
            unit = "pixels"

        print(f"\nAverage nearest-neighbor distance: {avg_distance_mm:.2f} {unit}")

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(distances_mm, bins=20, color='mediumslateblue', edgecolor='black')
        plt.title(f'Nearest-Neighbor Distance Distribution ({unit})')
        plt.xlabel(f'Distance ({unit})')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough dots to compute distances.")

    
    # Show final result
    final_result = image.copy()
    cv2.drawContours(final_result, optimized_contours, -1, (0, 255, 0), 2)
    for cx, cy in optimized_centroids:
        cv2.circle(final_result, (cx, cy), 5, (0, 0, 255), 2)
    
    cv2.imshow("Final Optimized Detection", final_result)
    cv2.imshow("Optimized Mask", optimized_mask)
    
    print(f"\nFinal result: {len(optimized_centroids)} purple dots detected")
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()