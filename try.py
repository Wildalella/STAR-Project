import cv2 as cv2
import numpy as np

SCALE_BAR_MM = 2.0


def nothing(x):
    pass

img = cv2.imread("image.png")




def create_hsv_trackbar_window(window_name='HSV Trackbars'):
    cv2.namedWindow(window_name)
    cv2.createTrackbar('H Lower', window_name, 125, 179, nothing)
    cv2.createTrackbar('H Upper', window_name, 165, 179, nothing)
    cv2.createTrackbar('S Lower', window_name, 100, 255, nothing)
    cv2.createTrackbar('S Upper', window_name, 225, 255, nothing)
    cv2.createTrackbar('V Lower', window_name, 10, 255, nothing)
    cv2.createTrackbar('V Upper', window_name, 255, 255, nothing)


def find_dot_centroids(mask, min_area=5, max_area=5000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
    return centroids

def draw_centroids(image, centroids):
    for cx, cy in centroids:
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)


def calculate_distances(centroids):
    distances = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            pt1 = np.array(centroids[i])
            pt2 = np.array(centroids[j])
            dist = np.linalg.norm(pt1 - pt2)
            distances.append(dist)
    return distances
   
# --- Main Script --
def main():

    img = cv2.imread("image.png")
    if img is None:
        print("Error: Image not found.")
        return

    
# convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# === Live HSV Threshold Adjustment ===
create_hsv_trackbar_window()

while True:
    # Get current trackbar positions
    h_lower = cv2.getTrackbarPos('H Lower', 'HSV Trackbars')
    h_upper = cv2.getTrackbarPos('H Upper', 'HSV Trackbars')
    s_lower = cv2.getTrackbarPos('S Lower', 'HSV Trackbars')
    s_upper = cv2.getTrackbarPos('S Upper', 'HSV Trackbars')
    v_lower = cv2.getTrackbarPos('V Lower', 'HSV Trackbars')
    v_upper = cv2.getTrackbarPos('V Upper', 'HSV Trackbars')

    lower_hsv = np.array([h_lower, s_lower, v_lower])
    upper_hsv = np.array([h_upper, s_upper, v_upper])

    # Detect dots using current HSV range

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(img, img, mask=mask)

    centroids = find_dot_centroids(mask)
    output = result.copy()
    draw_centroids(output, centroids)

    distances = calculate_distances(centroids)
    print(f"Found {len(centroids)} dots. First 5 distances (pixels): {distances[:5]}")


    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered Result", result)
      

    key = cv2.waitKey(50)
    if key == 27 or key == ord('q'):  # ESC or 'q' to break
        break

   

cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
     

