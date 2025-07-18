# I implemnted track bar to find correct parametes for HSV

import cv2 as cv
import numpy as np

def nothing(x):
    pass

# load image
img = cv.imread("image.png")
if img is None:
    print("Error: Image not found.")
    exit()

# convert to hsv
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# create window
cv.namedWindow('Trackbars')
#lh 0, ls 0, lv 0. uh 179, us 90, uv 255

# create trackbars for lower and upper HSV
cv.createTrackbar('LH', 'Trackbars', 0, 179, nothing)
cv.createTrackbar('LS', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('LV', 'Trackbars', 0, 255, nothing)

cv.createTrackbar('UH', 'Trackbars', 179, 179, nothing)
cv.createTrackbar('US', 'Trackbars', 255, 255, nothing)
cv.createTrackbar('UV', 'Trackbars', 255, 255, nothing)

while True:

    lh = cv.getTrackbarPos('LH', 'Trackbars')
    ls = cv.getTrackbarPos('LS', 'Trackbars')
    lv = cv.getTrackbarPos('LV', 'Trackbars')
    uh = cv.getTrackbarPos('UH', 'Trackbars')
    us = cv.getTrackbarPos('US', 'Trackbars')
    uv = cv.getTrackbarPos('UV', 'Trackbars')

#us = 85 or 120
#uh = 179
#uv = 255
# everything else 0

    
    lower_hsv = np.array([lh, ls, lv])
    upper_hsv = np.array([uh, us, uv])

    # mask and filter 
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(img, img, mask=mask)

    # Show different windows
    cv.imshow('Original', img)
    cv.imshow('Mask', mask)
    cv.imshow('Isolated Color', result)

    # stop program
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break

cv.destroyAllWindows()
