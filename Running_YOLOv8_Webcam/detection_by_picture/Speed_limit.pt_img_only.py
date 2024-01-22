from ultralytics import YOLO
import cv2
import math
import os

# Set desired frame size
frame_width = 1280  # Adjust the width as needed
frame_height = 720  # Adjust the height as needed

# Specify input and output paths
input_path = "./Running_YOLOv8_Webcam/detection_by_picture/input_images_speed_sign"  
output_path = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_speed_sign"  

# Check if the input folder exists
if not os.path.exists(input_path):
    print(f"Error: Input folder '{input_path}' not found.")
    exit()

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Define class dictionary
class_dict = {
    0: 'Speed Limit 10', 1: 'Speed Limit 100', 2: 'Speed Limit 110', 3: 'Speed Limit 120',
    4: 'Speed Limit 20', 5: 'Speed Limit 30', 6: 'Speed Limit 40', 7: 'Speed Limit 50',
    8: 'Speed Limit 60', 9: 'Speed Limit 70', 10: 'Speed Limit 80', 11: 'Speed Limit 90', 12: 'Stop'
}

# Process each image in the input folder
for filename in os.listdir(input_path):
    img_path = os.path.join(input_path, filename)

    # Check if the file is an image
    if not (img_path.lower().endswith(".png") or img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg")):
        continue

    # Load the image
    img = cv2.imread(img_path)

    # Check if the image is loaded successfully
    if img is None:
        print(f"Error loading image: {img_path}")
        continue

    # Perform object detection
    model = YOLO("../YOLO-Weights/Speed_limit.pt")
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            conf = math.ceil((box.conf[0] * 500)) / 500
            cls = int(box.cls[0])
            class_name = class_dict[cls]
            label = f'{class_name}{conf}'

            t_size = cv2.getTextSize(label, 0, fontScale=2, thickness=2)[0]  # Increase fontScale
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 2, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  # Increase thickness and font size

            # Output detection results to the console
            print(f'Detected: {class_name} with confidence {conf}')

    # Save the processed image to the specified output path
    output_img_path = os.path.join(output_path, filename)
    cv2.imwrite(output_img_path, img)

    # Display the processed image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
