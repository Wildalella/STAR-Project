#Handles image annotation and cv2.imshow visualization logic.
import cv2
from config import DOT_CIRCLE_COLOR, DOT_CIRCLE_RADIUS, DOT_CIRCLE_THICKNESS

def show_dots(image, centroids, contours):
    output = image.copy()
    for cx, cy in centroids:
        cv2.circle(output, (cx, cy), DOT_CIRCLE_RADIUS, DOT_CIRCLE_COLOR, DOT_CIRCLE_THICKNESS)
    cv2.drawContours(output, contours, -1, (0, 0, 255), 1)
    cv2.imshow("Detected Dots", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
