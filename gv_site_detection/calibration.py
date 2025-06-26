#Handles scale bar reading and mm/px conversion.
import cv2
import numpy as np
from config import SCALE_BAR_ROI_COORDS, SCALE_BAR_MM

def calibrate_scale(image):
    y1, y2, x1, x2 = SCALE_BAR_ROI_COORDS
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No scale bar found")
    scale_bar = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(scale_bar)
    return w / SCALE_BAR_MM
