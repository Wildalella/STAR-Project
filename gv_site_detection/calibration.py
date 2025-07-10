#Handles scale bar reading and mm/px conversion.
import cv2
import numpy as np
from config import SCALE_BAR_ROI_COORDS, SCALE_BAR_MM

# --- Helper Functions ---
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
