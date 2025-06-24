import cv2 as cv
import numpy as np

def nothing(x):
    pass

# Load the image
img = cv.imread("image.png")
if img is None:
    print("Error: Image not found.")
    exit()

# Convert to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Create window
cv.namedWindow('Trackbars')

# Create 6 trackbars for lower and upper HSV
cv.createTrackbar('LH', 'Trackbars', 0, 179, nothing)
cv.createTrackbar('LS', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('LV', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('UH', 'Trackbars', 179, 179, nothing)
cv.createTrackbar('US', 'Trackbars', 255, 255, nothing)
cv.createTrackbar('UV', 'Trackbars', 255, 255, nothing)

while True:
    # Get current positions of all trackbars
    lh = cv.getTrackbarPos('LH', 'Trackbars')
    ls = cv.getTrackbarPos('LS', 'Trackbars')
    lv = cv.getTrackbarPos('LV', 'Trackbars')
    uh = cv.getTrackbarPos('UH', 'Trackbars')
    us = cv.getTrackbarPos('US', 'Trackbars')
    uv = cv.getTrackbarPos('UV', 'Trackbars')

#us = 85 or 120
#uh =179
#uv = 255
#everything else 0

    # Define HSV range from trackbars
    lower_hsv = np.array([lh, ls, lv])
    upper_hsv = np.array([uh, us, uv])

    # Create a mask and filter the image
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(img, img, mask=mask)

    # Show original and result
    cv.imshow('Original', img)
    cv.imshow('Mask', mask)
    cv.imshow('Isolated Color', result)

    # Break on ESC
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break

cv.destroyAllWindows()
