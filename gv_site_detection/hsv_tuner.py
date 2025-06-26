#Contains the trackbar code and returns selected HSV values.
import cv2
import numpy as np

def setup_hsv_trackbars(window='Trackbars'):
    cv2.namedWindow(window)
    cv2.createTrackbar('LH', window, 125, 179, lambda x: None)
    cv2.createTrackbar('LS', window, 0, 255, lambda x: None)
    cv2.createTrackbar('LV', window, 0, 255, lambda x: None)
    cv2.createTrackbar('UH', window, 160, 179, lambda x: None)
    cv2.createTrackbar('US', window, 120, 255, lambda x: None)
    cv2.createTrackbar('UV', window, 255, 255, lambda x: None)

def get_hsv_from_trackbars(window='Trackbars'):
    lh = cv2.getTrackbarPos('LH', window)
    ls = cv2.getTrackbarPos('LS', window)
    lv = cv2.getTrackbarPos('LV', window)
    uh = cv2.getTrackbarPos('UH', window)
    us = cv2.getTrackbarPos('US', window)
    uv = cv2.getTrackbarPos('UV', window)
    return np.array([lh, ls, lv]), np.array([uh, us, uv])

def get_hsv_thresholds(image):
    setup_hsv_trackbars()
    while True:
        lower, upper = get_hsv_from_trackbars()
        mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("Mask", mask)
        cv2.imshow("Filtered", result)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return lower, upper
