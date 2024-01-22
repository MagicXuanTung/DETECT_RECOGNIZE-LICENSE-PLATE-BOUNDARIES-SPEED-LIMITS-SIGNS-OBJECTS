from ultralytics import YOLO
import cv2
import easyocr

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Set the desired window size (adjust these values as needed)
window_width = 800
window_height = 600

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", window_width, window_height)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/license_plate_detector.pt")

# Update class name for license plate
classNames = ["license_plate"]

# Initialize EasyOCR with the desired language (replace 'en' with the appropriate language code)
reader = easyocr.Reader(['en'])

while True:
    success, img = cap.read()

    # Doing detections using YOLOv8 frame by frame
    results = model_yolo(img, stream=True)

    # Once we have the results we will loop through them and extract license plate bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the detected object is a license plate
            cls = int(box.cls[0])
            if classNames[cls] == "license_plate":
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Crop the license plate region
                plate_region = img[y1:y2, x1:x2]

                # Perform OCR on the license plate region
                results_ocr = reader.readtext(plate_region)

                if results_ocr:
                    # Concatenate lines of the license plate text into a single string
                    license_plate_text = ' '.join(result[1] for result in results_ocr)

                    # Display the entire license plate text on a single line
                    cv2.putText(img, f'License Plate: {license_plate_text}', (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out.write(img)
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
