import cv2 
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('image.png')
im_resize = cv2.resize(im, None, None, 0.5, 0.5)

#Converts from BGR to HSV. Makes it filer by color
out = cv2.cvtColor(im_resize, cv2.COLOR_BGR2HSV)

#putting hue and saturation channels next to each other
stacked = np.hstack([out[...,0], out[...,1]])

#define thresholds for hue and saturation. Hue > 100 strong color, Saturation < 32(less colo intensity)
hue_thresh = 100
saturation_thresh = 32

#applying thresholds to hue and saturation channels. Creates a mask (true/false for each pixel) showing where both conditions are met.
thresh = np.logical_and(out[...,0] > hue_thresh, out[...,1] < saturation_thresh)

#Shows where the filtered pixels are.
cv2.imshow("Thresholded", 255*(thresh.astype(np.uint8)))
cv2.imshow("Hue & Saturation", stacked) 
cv2.imshow("Original Image", im_resize)


# Plotting histograms for hue and saturation channels, anaylze distribution of pixel intensities
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([im_resize], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])
plt.show()


hist = cv2.calcHist([out], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist, interpolation='nearest')
plt.title('Hue and Saturation Histogram')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.colorbar()
plt.show()






cv2.waitKey(0)
cv2.destroyAllWindows()