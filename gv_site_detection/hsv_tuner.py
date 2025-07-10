#Contains the trackbar code and returns selected HSV values.
import cv2
import numpy as np


def nothing(x):
    pass

def create_hsv_trackbar_window(window_name='HSV Trackbars'):
    cv2.namedWindow(window_name)
    cv2.createTrackbar('H Lower', window_name, 125, 179, nothing)
    cv2.createTrackbar('H Upper', window_name, 165, 179, nothing)
    cv2.createTrackbar('S Lower', window_name, 100, 255, nothing)
    cv2.createTrackbar('S Upper', window_name, 225, 255, nothing)
    cv2.createTrackbar('V Lower', window_name, 10, 255, nothing)
    cv2.createTrackbar('V Upper', window_name, 255, 255, nothing)