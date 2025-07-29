import cv2 
import numpy as np
from hueChannel import im, im_resize, out
hue_thresh = 100
saturation_thresh = 32

thresh = np.logical_and(out[...,0] > hue_thresh, out[...,1] < saturation_thresh)

cv2.imshow("Thresholded", 255*(thresh.astype(np.uint8)))
cv2.waitKey(0)
cv2.destroyAllWindows()