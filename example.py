import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy import stats  # For Kernel Density Estimate (though not used for main plots)
import csv
import math  # For math.pi and math.sqrt

# --- Configuration Parameters ---
IMAGE_PATH = 'image.png'

# Scale bar properties
SCALE_BAR_MM = 2.0
# IMPORTANT: Re-adjust these coordinates if they were changed for a different image version
SCALE_BAR_ROI_COORDS = (315, 330, 945, 1150)  # (y_start, y_end, x_start, x_end)

# --- Purple Color Range in HSV ---
#LOWER_PURPLE_HSV = np.array([125, 100, 10])
#UPPER_PURPLE_HSV = np.array([160, 255, 255])
# --- Purple Color Range in HSV ---
LOWER_PURPLE_HSV = np.array([125, 100, 10])
UPPER_PURPLE_HSV = np.array([165, 225, 255])



# Morphological operation parameters
MORPH_KERNEL_SIZE = (3, 3)
EROSION_ITERATIONS = 1 #1 
DILATION_ITERATIONS = 1 #1

# Dot filtering parameters
MIN_DOT_AREA_PX = 5 # 10
MAX_DOT_AREA_PX = 3000 #changed 1000

# Histogram parameters
HIST_MIN_MM = 0.1
HIST_MAX_MM = 2.0
HIST_BIN_WIDTH_MM = 0.1

# Heatmap parameters
HEATMAP_GRID_CELL_SIZE_MM = 1.0  # Each cell in the heatmap will be 1mm x 1mm

# Output file names
HISTOGRAM_IMAGE_PATH = 'dots_distance_histogram.png'
CSV_OUTPUT_PATH = 'histogram_data.csv'
DOT_SIZE_DISTRIBUTION_IMAGE_PATH = 'dot_sizes_distribution_microns.png'
DOT_DIAMETER_DISTRIBUTION_IMAGE_PATH = 'dot_diameters_distribution_microns.png'  # Updated plot filename
DOT_DENSITY_HEATMAP_IMAGE_PATH = 'dot_density_heatmap.png'

# Visualization parameters for detected dots
SHOW_DETECTED_DOTS_IMAGE = True
DOT_CIRCLE_RADIUS = 5
DOT_CIRCLE_COLOR = (0, 255, 0)  # Green in BGR
DOT_CIRCLE_THICKNESS = 1


# --- Helper Functions ---

def create_hsv_trackbars(window_name='HSV Trackbars'):
    cv2.namedWindow(window_name)

    # Hue ranges from 0–179 in OpenCV
    cv2.createTrackbar('H Low', window_name, 125, 179, lambda x: None)
    cv2.createTrackbar('H High', window_name, 165, 179, lambda x: None)

    # Saturation & Value: 0–255
    cv2.createTrackbar('S Low', window_name, 50, 255, lambda x: None)
    cv2.createTrackbar('S High', window_name, 225, 255, lambda x: None)

    cv2.createTrackbar('V Low', window_name, 50, 255, lambda x: None)
    cv2.createTrackbar('V High', window_name, 255, 255, lambda x: None)

def calibrate_scale(image, roi_coords, known_length_mm):
    y_start, y_end, x_start, x_end = roi_coords
    if not (0 <= y_start < y_end <= image.shape[0] and \
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
        print("Error: No contours found in scale bar ROI. Adjust ROI or threshold.")
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


def detect_dots_hsv(image, lower_hsv, upper_hsv, min_area, max_area):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.erode(mask, kernel, iterations=EROSION_ITERATIONS)
    mask = cv2.dilate(mask, kernel, iterations=DILATION_ITERATIONS)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_centroids = []
    valid_dot_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dot_centroids.append((cx, cy))
                valid_dot_contours.append(contour)
    print(f"Detected {len(dot_centroids)} dots using HSV thresholding.")
    return dot_centroids, valid_dot_contours


# --- Main Script ---
def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return

    original_image_height_px, original_image_width_px = image.shape[:2]
    print(f"Image loaded: {original_image_width_px}x{original_image_height_px} pixels.")

    pixels_per_mm = calibrate_scale(image, SCALE_BAR_ROI_COORDS, SCALE_BAR_MM)
    if pixels_per_mm is None:
        print("Exiting due to scale calibration failure.")
        cv2.destroyAllWindows()
        return

    mm_per_pixel = 1.0 / pixels_per_mm
    microns_per_mm = 1000.0
    microns_per_pixel = mm_per_pixel * microns_per_mm
    microns_per_pixel_sq = microns_per_pixel ** 2

    # Image dimensions in mm
    image_width_mm = original_image_width_px * mm_per_pixel
    image_height_mm = original_image_height_px * mm_per_pixel

    # Detect dots using HSV thresholding
    dot_centroids_px, detected_dot_contours = detect_dots_hsv(
        image, LOWER_PURPLE_HSV, UPPER_PURPLE_HSV,
        MIN_DOT_AREA_PX, MAX_DOT_AREA_PX
    )

    if SHOW_DETECTED_DOTS_IMAGE:
        image_with_dots = image.copy()
        cv2.drawContours(image_with_dots, detected_dot_contours, -1, (0, 0, 255), 1)
        for cx, cy in dot_centroids_px:
            cv2.circle(image_with_dots, (cx, cy), DOT_CIRCLE_RADIUS, DOT_CIRCLE_COLOR, DOT_CIRCLE_THICKNESS)

        cv2.imshow("Identified Dots on Original Image (HSV)", image_with_dots)
        print("Displaying image with identified dots. Press any key in the image window to continue...")
        cv2.waitKey(0)

    if len(dot_centroids_px) < 1:
        print("Not enough dots detected for analysis.")
        cv2.destroyAllWindows()
        return

    # --- Dot Density Heatmap ---
    if dot_centroids_px:
        # Determine grid dimensions
        grid_cols = math.ceil(image_width_mm / HEATMAP_GRID_CELL_SIZE_MM)
        grid_rows = math.ceil(image_height_mm / HEATMAP_GRID_CELL_SIZE_MM)

        density_grid = np.zeros((grid_rows, grid_cols), dtype=int)

        for cx_px, cy_px in dot_centroids_px:
            # Convert dot centroid to mm coordinates
            cx_mm = cx_px * mm_per_pixel
            cy_mm = cy_px * mm_per_pixel

            # Determine which grid cell the dot falls into
            col_idx = math.floor(cx_mm / HEATMAP_GRID_CELL_SIZE_MM)
            row_idx = math.floor(cy_mm / HEATMAP_GRID_CELL_SIZE_MM)

            # Ensure indices are within bounds
            if 0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols:
                density_grid[row_idx, col_idx] += 1

        plt.figure(figsize=(10, 8))
        plt.imshow(density_grid, cmap='hot', interpolation='nearest',
                   origin='upper', extent=[0, image_width_mm, image_height_mm, 0])
        plt.colorbar(label=f'Dots per {HEATMAP_GRID_CELL_SIZE_MM}x{HEATMAP_GRID_CELL_SIZE_MM} mm²')
        plt.title('Purple Dot Density Heatmap')
        plt.xlabel('Image Width (mm)')
        plt.ylabel('Image Height (mm)')
        plt.tight_layout()
        plt.savefig(DOT_DENSITY_HEATMAP_IMAGE_PATH)
        print(f"Dot density heatmap saved to {DOT_DENSITY_HEATMAP_IMAGE_PATH}")

    # --- Dot Size (Area) Analysis ---
    dot_areas_px = [cv2.contourArea(c) for c in detected_dot_contours]
    dot_areas_microns2 = [area_px * microns_per_pixel_sq for area_px in dot_areas_px]

    if dot_areas_microns2:
        average_dot_size_microns2 = np.mean(dot_areas_microns2)
        print(f"Average dot area: {average_dot_size_microns2:.2f} µm^2")

        plt.figure(figsize=(10, 6))
        n_bins_size = 20
        plt.hist(dot_areas_microns2, bins=n_bins_size, density=False, alpha=0.6, color='g', edgecolor='black',
                 label='Dot Area Histogram')

        plt.title('Distribution of Purple Dot Areas')
        plt.xlabel('Dot Area (µm²)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(DOT_SIZE_DISTRIBUTION_IMAGE_PATH)
        print(f"Dot area distribution plot saved to {DOT_SIZE_DISTRIBUTION_IMAGE_PATH}")
    else:
        print("No dot areas to analyze for size distribution.")

    # --- Dot Diameter Analysis ---
    # Calculate radii from areas: Area = pi * r^2  => r = sqrt(Area / pi)
    # Diameter = 2 * r
    dot_diameters_microns = [2 * math.sqrt(area / math.pi) for area in dot_areas_microns2 if area > 0]

    if dot_diameters_microns:
        average_dot_diameter_microns = np.mean(dot_diameters_microns)
        print(f"Average dot diameter: {average_dot_diameter_microns:.2f} µm")

        plt.figure(figsize=(10, 6))
        n_bins_diameter = 20
        plt.hist(dot_diameters_microns, bins=n_bins_diameter, density=False, alpha=0.6, color='orange',
                 edgecolor='black', label='Dot Diameter Histogram')

        plt.title('Distribution of Purple Dot Diameters')
        plt.xlabel('Dot Diameter (µm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(DOT_DIAMETER_DISTRIBUTION_IMAGE_PATH)
        print(f"Dot diameter distribution plot saved to {DOT_DIAMETER_DISTRIBUTION_IMAGE_PATH}")
    else:
        print("No dot diameters to analyze for diameter distribution.")

    # --- Nearest Neighbor Distance Analysis ---
    if len(dot_centroids_px) < 2:
        print("Not enough dots detected to calculate distances (need at least 2).")
        cv2.destroyAllWindows()
        return

    centroids_array_px = np.array(dot_centroids_px)
    if centroids_array_px.size == 0 or centroids_array_px.shape[0] < 2:
        print("No valid dot centroids array for distance calculation.")
        cv2.destroyAllWindows()
        return

    kdtree = KDTree(centroids_array_px)
    distances_px, _ = kdtree.query(centroids_array_px, k=2)
    nearest_neighbor_distances_px = distances_px[:, 1]
    nearest_neighbor_distances_mm = nearest_neighbor_distances_px * mm_per_pixel

    filtered_distances_mm = nearest_neighbor_distances_mm[
        (nearest_neighbor_distances_mm >= HIST_MIN_MM) &
        (nearest_neighbor_distances_mm <= HIST_MAX_MM)
        ]

    if filtered_distances_mm.size == 0:
        print(f"No dot distances found in the range {HIST_MIN_MM}mm to {HIST_MAX_MM}mm.")
    else:
        print(f"Found {len(filtered_distances_mm)} nearest neighbor distances in the specified range.")

        plt.figure(figsize=(10, 6))
        bins = np.arange(HIST_MIN_MM, HIST_MAX_MM + HIST_BIN_WIDTH_MM, HIST_BIN_WIDTH_MM)
        counts, bin_edges = np.histogram(filtered_distances_mm, bins=bins)
        plt.hist(filtered_distances_mm, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
        plt.title('Histogram of Nearest Neighbor Distances (HSV Detections)')
        plt.xlabel('Distance (mm)')
        plt.ylabel('Frequency')
        plt.xticks(bins, rotation=45, ha="right")
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(HISTOGRAM_IMAGE_PATH)
        print(f"Distance histogram saved to {HISTOGRAM_IMAGE_PATH}")

        with open(CSV_OUTPUT_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['bin_start_mm', 'bin_end_mm', 'frequency'])
            for i in range(len(counts)):
                writer.writerow([f"{bin_edges[i]:.2f}", f"{bin_edges[i + 1]:.2f}", counts[i]])
        print(f"Distance histogram data saved to {CSV_OUTPUT_PATH}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()