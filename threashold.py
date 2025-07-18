import cv2
import numpy as np

image = cv2.imread('image.png', 0)
th, res = cv2.threshold(image, None, 255, cv2.THRESH_TRIANGLE)
#th, res2 = cv2.threshold(image, None, 255, cv2.THRESH_TRIANGLE)


# cv2.THRESH_TRIANGLE


 
# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector()
# Detect blobs.
params = cv2.SimpleBlobDetector_Params()
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,225), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow('Original', image)
cv2.imshow("Thresholded Image", res )
#cv2.imshow("Thresholded Image", res2 )


cv2.waitKey(0)
cv2.destroyAllWindows()


 


 
