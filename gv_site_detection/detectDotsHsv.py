#HSV parameters starting to get implemented. I am going to try to merge example.py(orginal code) to this file

import cv2
import numpy as np


def nothing(x):
    pass

# Load the image
img = cv2.imread("image.png")
if img is None:
    print("Error: Image not found.")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    # Define HSV range from trackbars
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([179, 85, 255])

        # Create a mask and filter the image
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(img, img, mask=mask)

        # Show original and result
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Isolated Color', result)


    # Break on ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()




 #
#Runs the HSV-based blob detection logic.
#def detect_dots_hsv(image, lower, upper, min_area, max_area):
 #   mask = cv22.inRange(cv22.cv2tColor(image, cv22.COLOR_BGR2HSV), lower, upper)
  #  mask = cv22.morphologyEx(mask, cv22.MORPH_OPEN, (3, 3))
    #  contours, _ = cv22.findContours(mask, cv22.RETR_EXTERNAL, cv22.CHAIN_APPROX_SIMPLE)
    #  centroids = []
  #    valid = []
   #   for c in contours:
     #     if min_area < cv22.contourArea(c) < max_area:
      #        M = cv22.moments(c)
      #        if M["m00"]:
       #           cx = int(M["m10"]/M["m00"])
       #           cy = int(M["m01"]/M["m00"])
          #        centroids.append((cx, cy))
          #        valid.append(c)
    #  return centroids, valid
