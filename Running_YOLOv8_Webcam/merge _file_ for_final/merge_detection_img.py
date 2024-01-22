import os
from ultralytics import YOLO
import cv2
import math
import numpy as np
import easyocr

# Set input and output directories
image_directory = "./Running_YOLOv8_Webcam/merge _file_ for_final/input_merge"
output_directory = "./Running_YOLOv8_Webcam/merge _file_ for_final/output_merge"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

model_yolo = YOLO("../YOLO-Weights/yolov8n.pt")
desired_classes_yolo = ["person", "car", "truck", "train", "bicycle"]

count_dict_yolo = {class_name: 0 for class_name in desired_classes_yolo}

# Define object-specific variables
focal = 260
width = 4

# Find the distance from the camera
def get_dist(rectangle_params):
    rect_width = rectangle_params[1][0]
    dist = (width * focal) / rect_width / 100  # Convert to meters
    return dist

# Update the path to your YOLO license plate detection model
model_license_plate = YOLO("../YOLO-Weights/license_plate_detector.pt")

# Update class name for license plate
classNames_license_plate = ["license_plate"]

# Initialize EasyOCR with the desired language (replace 'en' with the appropriate language code)
reader = easyocr.Reader(['en'])

# Process each image in the input directory
for filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, filename)
    img = cv2.imread(image_path)

    # Doing detections using YOLOv8 for each image
    results_yolo = model_yolo(img)

    # Reset count for each YOLO frame
    count_dict_yolo = {class_name: 0 for class_name in desired_classes_yolo}

    # Loop through each of the YOLO results
    for r in results_yolo:
        boxes = r.boxes

        # Loop through each YOLO bounding box
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the detected class is in the desired classes
            cls = int(box.cls[0])
            class_name = model_yolo.names[cls]
            if class_name in desired_classes_yolo:
                # Increment count for the detected class
                count_dict_yolo[class_name] += 1

                # Choose a color based on the class (red for license plate, magenta for others)
                color = (0, 0, 255) if class_name == "license_plate" else (255, 0, 255)

                # Draw bounding box on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Draw label on the original image
                conf = box.conf[0]
                label = f'{class_name}{conf:.2f}'  # Format the confidence to two decimal places
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # Get distance and display it under the bounding box
                rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x2, y1], [x1, y2]]))
                distance = get_dist(rect)
                dist_text = f'Distance: {distance:.2f} meters'
                cv2.putText(img, dist_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw text line with the count of each YOLO class
    count_text_yolo = ', '.join([f'{class_name}: {count}' for class_name, count in count_dict_yolo.items()])
    cv2.putText(img, count_text_yolo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Doing detections using YOLOv8 for license plate in the detected objects
    results_license_plate = model_license_plate(img)

    # Loop through the license plate detection results
    for r in results_license_plate:
        boxes_license_plate = r.boxes
        for box_license_plate in boxes_license_plate:
            x1_lp, y1_lp, x2_lp, y2_lp = box_license_plate.xyxy[0]
            x1_lp, y1_lp, x2_lp, y2_lp = int(x1_lp), int(y1_lp), int(x2_lp), int(y2_lp)

            # Check if the detected object is a license plate
            cls_lp = int(box_license_plate.cls[0])
            if classNames_license_plate[cls_lp] == "license_plate":
                # Draw bounding box on the original image (using red color)
                cv2.rectangle(img, (x1_lp, y1_lp), (x2_lp, y2_lp), (0, 0, 255), 3)

                # Crop the license plate region
                plate_region = img[y1_lp:y2_lp, x1_lp:x2_lp]

                # Print coordinates for debugging
                print(f"License Plate Coordinates: ({x1_lp}, {y1_lp}, {x2_lp}, {y2_lp})")

                # Perform OCR on the license plate region
                results_ocr = reader.readtext(plate_region)

                # Print OCR results for debugging
                print("OCR Results:", results_ocr)

                if results_ocr:
                    # Concatenate lines of the license plate text into a single string
                    license_plate_text = ' '.join(result[1] for result in results_ocr)

                    # Display the entire license plate text on a single line
                    cv2.putText(img, f'License Plate: {license_plate_text}', (x1_lp, y1_lp - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the processed image
    cv2.imshow("Image", img)
    
    # Wait for a key event to close the window
    cv2.waitKey(0)

# Close all windows at the end
cv2.destroyAllWindows()
