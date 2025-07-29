import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize image
im = cv2.imread('image.png')
im_resize = cv2.resize(im, None, None, 0.5, 0.5)

# Convert to HSV
hsv = cv2.cvtColor(im_resize, cv2.COLOR_BGR2HSV)

# Set HSV thresholds to isolate stain color (tweak as needed)
hue_thresh = 100
saturation_thresh = 32
mask = np.logical_and(hsv[..., 0] > hue_thresh, hsv[..., 1] < saturation_thresh).astype(np.uint8) * 255

# Find contours (each contour is a stain)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get areas of stains
areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]  # filter out any tiny noise

# Draw contours for visualization
im_contours = im_resize.copy()
cv2.drawContours(im_contours, contours, -1, (0, 255, 0), 2)

# Show images
cv2.imshow("Original", im_resize)
cv2.imshow("Mask", mask)
cv2.imshow("Contours", im_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot histogram
plt.hist(areas, bins=20, color='blue', edgecolor='black')
plt.title("Histogram of Stain Sizes")
plt.xlabel("Area (pixels)")
plt.ylabel("Count")
plt.show()
