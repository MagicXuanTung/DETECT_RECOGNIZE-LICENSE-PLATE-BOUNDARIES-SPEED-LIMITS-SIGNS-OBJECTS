import numpy as np
import cv2

# Define object-specific variables
focal = 450
width = 4

# Find the distance from the camera
def get_dist(rectangle_params, image):
    # Extract the width of the rectangle
    rect_width = rectangle_params[1][0]

    # Calculate distance
    dist = (width * focal) / rect_width

    # Write on the image
    image = cv2.putText(image, f'Distance from Camera: {dist:.2f} cm', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Extract frames
cap = cv2.VideoCapture(0)

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')

cv2.namedWindow('Object Dist Measure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure', 700, 600)

# Loop to capture video frames
while True:
    ret, img = cap.read()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Predefined mask for green color detection
    lower = np.array([37, 51, 24])
    upper = np.array([83, 104, 131])
    mask = cv2.inRange(hsv_img, lower, upper)

    # Remove extra garbage from the image
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find the contours
    contours, _ = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for cnt in contours:
        # Check for contour area
        if 100 < cv2.contourArea(cnt) < 306000:
            # Draw a rectangle on the contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], -1, (255, 0, 0), 3)

            img = get_dist(rect, img)

    cv2.imshow('Object Dist Measure', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
