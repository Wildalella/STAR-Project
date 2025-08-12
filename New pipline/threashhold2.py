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

# --- Tight exclusion zones around black symbols only ---
EXCLUSION_ZONES = [
    # "f" symbol (tight box around just the letter)
    (315, 335, 1125, 1145),  # (y_start, y_end, x_start, x_end)
    
    # "2" in "2 mm" (tight box)
    (340, 360, 1125, 1140),  
    
    # "mm" text (tight box)
    (340, 360, 1140, 1170),
    
    # Black scale bar line (tight box around just the line)
    (330, 340, 945, 1125),
]


def is_in_exclusion_zone(x, y, exclusion_zones):
    """Check if a point (x, y) falls within any exclusion zone"""
    for y_start, y_end, x_start, x_end in exclusion_zones:
        if x_start <= x <= x_end and y_start <= y <= y_end:
            return True
    return False


def filter_by_color_intensity(contours, image, exclusion_zones=None):
    """
    Additional filter to remove very dark/black symbols while keeping purple stains.
    Uses color intensity to distinguish between black text and purple dots.
    """
    if exclusion_zones is None:
        exclusion_zones = []
    
    filtered_contours = []
    
    for contour in contours:
        # Get contour properties
        area = cv2.contourArea(contour)
        
        # Skip very small or very large contours
        if area < 10 or area > 2000:
            continue
        
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Check if centroid is in tight exclusion zone (black symbols)
        if is_in_exclusion_zone(cx, cy, exclusion_zones):
            continue
        
        # Create mask for this contour to analyze its color
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Get mean color values in the contour region
        mean_bgr = cv2.mean(image, mask=mask)[:3]  # Get B, G, R values
        mean_brightness = np.mean(mean_bgr)
        
        # Convert to HSV for better color analysis
        # Create a small patch around centroid for color sampling
        patch_size = 5
        y_start = max(0, cy - patch_size)
        y_end = min(image.shape[0], cy + patch_size)
        x_start = max(0, cx - patch_size)
        x_end = min(image.shape[1], cx + patch_size)
        
        patch = image[y_start:y_end, x_start:x_end]
        if patch.size > 0:
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mean_saturation = np.mean(hsv_patch[:, :, 1])
        else:
            mean_saturation = 0
        
        # Filter criteria:
        # 1. Not too dark (removes black text) - brightness > 30
        # 2. Has some color saturation (removes gray/black) - saturation > 20
        # 3. Not in exclusion zones
        if (mean_brightness > 30 and mean_saturation > 20):
            filtered_contours.append(contour)
    
    return filtered_contours


def create_optimized_detection_function(image):
    """Create an optimized detection function based on your specific image"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
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
    
    # Create exclusion mask to remove known text/scale areas
    exclusion_mask = np.ones(combined_mask.shape, dtype=np.uint8) * 255
    for y_start, y_end, x_start, x_end in EXCLUSION_ZONES:
        # Ensure coordinates are within image bounds
        y_start = max(0, y_start)
        y_end = min(image.shape[0], y_end)
        x_start = max(0, x_start)
        x_end = min(image.shape[1], x_end)
        exclusion_mask[y_start:y_end, x_start:x_end] = 0
    
    # Apply exclusion mask
    combined_mask = cv2.bitwise_and(combined_mask, exclusion_mask)
    
    # Gentle cleanup
    kernel = np.ones((2, 2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask


def detect_contours_with_shape_filtering(mask, image, method_name):
    """Detect contours with color and location-based filtering to exclude only black symbols"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], []
    
    # Apply color intensity and tight location filtering
    filtered_contours = filter_by_color_intensity(contours, image, EXCLUSION_ZONES)
    
    if not filtered_contours:
        print(f"{method_name}: No valid dots found after filtering")
        return [], []
    
    # Calculate area statistics on filtered contours
    areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    
    # More flexible filtering based on area
    min_area = max(10, mean_area - 2 * std_area)
    max_area = min(2000, mean_area + 3 * std_area)

    dot_centroids = []
    valid_contours = []
    
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dot_centroids.append((cx, cy))
                valid_contours.append(contour)
    
    print(f"{method_name}: Detected {len(dot_centroids)} dots after filtering (area range: {min_area:.1f} - {max_area:.1f})")
    return dot_centroids, valid_contours


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


def visualize_exclusion_zones(image, exclusion_zones):
    """Visualize exclusion zones on the image for debugging"""
    viz_image = image.copy()
    for y_start, y_end, x_start, x_end in exclusion_zones:
        # Draw rectangle for exclusion zone
        cv2.rectangle(viz_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        # Add text label
        cv2.putText(viz_image, "EXCLUDED", (x_start, y_start-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return viz_image


def main():
    """Enhanced main function with text/scale bar filtering"""
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

    print("\n=== USING FILTERED DETECTION (EXCLUDING TEXT & SCALE BAR) ===")
    
    # Show exclusion zones
    exclusion_viz = visualize_exclusion_zones(image, EXCLUSION_ZONES)
    cv2.imshow("Exclusion Zones (Blue Rectangles)", exclusion_viz)
    
    # Apply optimized detection with filtering
    filtered_mask = create_optimized_detection_function(image)
    filtered_centroids, filtered_contours = detect_contours_with_shape_filtering(
        filtered_mask, image, "Filtered Detection"
    )

    # Distance analysis between centroids
    if len(filtered_centroids) >= 2:
        tree = KDTree(filtered_centroids)
        distances = []

        for i, point in enumerate(filtered_centroids):
            dists, idxs = tree.query(point, k=2)
            nearest_dist = dists[1]
            distances.append(nearest_dist)

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
        plt.title(f'Purple Dot Distance Distribution ({unit})')
        plt.xlabel(f'Distance ({unit})')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough dots to compute distances.")

    # Show final result
    final_result = image.copy()
    cv2.drawContours(final_result, filtered_contours, -1, (0, 255, 0), 2)
    for cx, cy in filtered_centroids:
        cv2.circle(final_result, (cx, cy), 5, (0, 0, 255), 2)
    
    cv2.imshow("Filtered Purple Dot Detection", final_result)
    cv2.imshow("Filtered Mask", filtered_mask)
    
    print(f"\nFiltered result: {len(filtered_centroids)} purple dots detected (black symbols excluded)")
    print("Blue rectangles show tight exclusion zones around black symbols only")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()