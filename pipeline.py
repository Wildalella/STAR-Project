import cv2
import numpy as np

#Create folder that does threshhold, hsv, and blob detction next( 7/19/2025)

image = cv2.imread('image.png', 0)
image_color = cv2.imread('image.png')

# Thresholding
th, res = cv2.threshold(image, None, 255, cv2.THRESH_TRIANGLE)



# HSV filtering (if your dots have a specific color)
hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

# Example: filter for dark colors (adjust valuea as needed)
lower = np.array([0, 0, 0])
upper = np.array([179, 85, 255])
#lh 0, ls 0, lv 0. uh 179, us 90, uv 255

mask_hsv = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(image_color, image_color, mask=mask_hsv)

combined_mask = cv2.bitwise_and(res, mask_hsv)

# Detect blobs.
params = cv2.SimpleBlobDetector_Params()
detector = cv2.SimpleBlobDetector_create(params)
keypoints_thresh = detector.detect(res)
keypoints_hsv = detector.detect(mask_hsv)
keypoints = detector.detect(combined_mask)


# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,225), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_hsv = cv2.drawKeypoints(image_color, keypoints_hsv, np.array([]), (0,0,225), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_thresh = cv2.drawKeypoints(image_color, keypoints_thresh, np.array([]), (0,0,225), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

cv2.imshow('Thresh Original', image)
cv2.imshow("Thresholded Image", res )

cv2.imshow('HSV Mask', mask_hsv)
cv2.imshow('Isolated Color', result)

cv2.imshow('Combined Mask', combined_mask)





cv2.waitKey(0)
cv2.destroyAllWindows()

