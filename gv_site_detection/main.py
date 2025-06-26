#from calibration import calibrate_scale
from hsv_tuner import get_hsv_thresholds
from detection import detect_dots_hsv
from analysis import analyze_dots
from visualization import show_dots
from config import IMAGE_PATH

def main():
    image = cv2.imread(IMAGE_PATH)
    pixels_per_mm = calibrate_scale(image)
    lower_hsv, upper_hsv = get_hsv_thresholds(image)
    dot_centroids, contours = detect_dots_hsv(image, lower_hsv, upper_hsv)
    analyze_dots(dot_centroids, contours, pixels_per_mm, image.shape)
    show_dots(image, dot_centroids, contours)

if __name__ == "__main__":
    main()
