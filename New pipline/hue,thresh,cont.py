import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# Constants
IMAGE_PATH = 'image.png'
RESIZE_SCALE = 0.5
HUE_THRESHOLD = 100
SATURATION_THRESHOLD = 32
MIN_CONTOUR_AREA = 100  # Minimum area for contour detection


def load_and_resize_image(filepath: str, scale: float = RESIZE_SCALE) -> np.ndarray:
    """
    Load and resize an image from file.
    
    Args:
        filepath: Path to the image file
        scale: Resize scale factor
        
    Returns:
        Resized image as numpy array
        
    Raises:
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not load image: {filepath}")
    
    return cv2.resize(image, None, None, scale, scale)


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSV color space.
    
    Args:
        image: BGR image
        
    Returns:
        HSV image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def create_color_mask(hsv_image: np.ndarray, 
                     hue_thresh: int = HUE_THRESHOLD, 
                     sat_thresh: int = SATURATION_THRESHOLD) -> np.ndarray:
    """
    Create a binary mask based on HSV thresholds.
    
    Args:
        hsv_image: HSV color image
        hue_thresh: Minimum hue threshold
        sat_thresh: Maximum saturation threshold
        
    Returns:
        Binary mask as uint8 array
    """
    # Create mask where hue > threshold AND saturation < threshold
    mask = np.logical_and(
        hsv_image[..., 0] > hue_thresh,  # Hue channel
        hsv_image[..., 1] < sat_thresh   # Saturation channel
    )
    return (mask * 255).astype(np.uint8)


def find_and_draw_contours(mask: np.ndarray, 
                          original_image: np.ndarray,
                          min_area: int = MIN_CONTOUR_AREA) -> Tuple[np.ndarray, List]:
    """
    Find contours in the mask and draw them on the original image.
    
    Args:
        mask: Binary mask
        original_image: Original image to draw contours on
        min_area: Minimum contour area to consider
        
    Returns:
        Tuple of (image with contours, list of contours)
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create a copy of the original image to draw on
    contour_image = original_image.copy()
    
    # Draw contours
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
    
    # Draw bounding rectangles for each contour
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return contour_image, filtered_contours


def display_images(original: np.ndarray, 
                  hsv_channels: np.ndarray, 
                  mask: np.ndarray, 
                  contours: np.ndarray) -> None:
    """
    Display all processed images in OpenCV windows.
    
    Args:
        original: Original resized image
        hsv_channels: Stacked hue and saturation channels
        mask: Binary threshold mask
        contours: Image with contours drawn
    """
    cv2.imshow("1. Original Image", original)
    cv2.imshow("2. Hue & Saturation Channels", hsv_channels)
    cv2.imshow("3. Color Mask", mask)
    cv2.imshow("4. Detected Contours", contours)


def plot_color_histograms(bgr_image: np.ndarray, hsv_image: np.ndarray) -> None:
    """
    Plot BGR and HSV histograms for analysis.
    
    Args:
        bgr_image: Original BGR image
        hsv_image: HSV converted image
    """
    # BGR histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('BGR Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([bgr_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f'{color.upper()} channel')
    
    plt.xlim([0, 256])
    plt.legend()
    
    # HSV 2D histogram
    plt.subplot(1, 2, 2)
    hist_2d = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist_2d, interpolation='nearest', aspect='auto')
    plt.title('Hue vs Saturation Histogram')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.colorbar(label='Pixel Count')
    
    plt.tight_layout()
    plt.show()


def print_contour_info(contours: List, image_shape: Tuple) -> None:
    """
    Print information about detected contours.
    
    Args:
        contours: List of detected contours
        image_shape: Shape of the processed image
    """
    print(f"\n--- Contour Detection Results ---")
    print(f"Image dimensions: {image_shape[1]}x{image_shape[0]}")
    print(f"Total contours found: {len(contours)}")
    
    if contours:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        print(f"Contour areas: {[int(area) for area in areas]}")
        print(f"Largest contour area: {int(max(areas))}")
        print(f"Total filtered area: {int(sum(areas))} pixels")
    else:
        print("No contours detected with current thresholds.")


def main() -> None:
    """
    Main function to execute the color filtering and contour detection pipeline.
    """
    try:
        # Load and preprocess image
        print("Loading and preprocessing image...")
        original_image = load_and_resize_image(IMAGE_PATH)
        hsv_image = convert_to_hsv(original_image)
        
        # Create visualization of HSV channels
        hue_saturation_stack = np.hstack([hsv_image[..., 0], hsv_image[..., 1]])
        
        # Apply color filtering
        print("Applying HSV color filtering...")
        color_mask = create_color_mask(hsv_image)
        
        # Find and draw contours
        print("Detecting contours...")
        contour_image, detected_contours = find_and_draw_contours(color_mask, original_image)
        
        # Display results
        display_images(original_image, hue_saturation_stack, color_mask, contour_image)
        
        # Print contour information
        print_contour_info(detected_contours, original_image.shape)
        
        # Plot histograms
        print("Generating histograms...")
        plot_color_histograms(original_image, hsv_image)
        
        # Wait for user input and cleanup
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()