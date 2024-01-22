import os
from ultralytics import YOLO
import cv2
import math
import time

cap = cv2.VideoCapture(0)

# Increase the window size
frame_width = 1280  # set your desired width
frame_height = 720  # set your desired height

cap.set(3, frame_width)
cap.set(4, frame_height)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO("../YOLO-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

output_directory = "./Running_YOLOv8_Webcam/detection_person_only/output_detected_persons"  # Create a folder to store the output images

# Set the time interval for saving images (in seconds)
save_interval = 5
start_time = time.time()

while True:
    success, img = cap.read()

    # Create a copy of the image to save without bounding boxes and labels
    img_without_boxes = img.copy()

    # Doing detections using YOLOv8 frame by frame
    results = model(img, stream=True)

    # Counter for bounding boxes
    bounding_box_count = 0

    # Loop through each of the bounding boxes in the results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the detected object is a person
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                # Increment the bounding box count
                bounding_box_count += 1

                # Draw bounding box on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Draw label on the original image
                conf = math.ceil((box.conf[0] * 500)) / 500
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Draw a text line with the count of bounding boxes
    cv2.putText(img, f'Person : {bounding_box_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the entire frame without bounding boxes and labels every 5 seconds
    elapsed_time = time.time() - start_time
    if bounding_box_count > 0 and elapsed_time >= save_interval:
        frame_filename = f'frame_{len(os.listdir(output_directory)) + 1}.jpg'
        cv2.imwrite(os.path.join(output_directory, frame_filename), img_without_boxes)
        print(f"Frame image saved: {frame_filename}")
        start_time = time.time()  # Reset the timer

    out.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

out.release()
cv2.destroyAllWindows()

