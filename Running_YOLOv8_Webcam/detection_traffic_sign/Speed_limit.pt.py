from ultralytics import YOLO
import cv2
import math

# Set desired frame size
frame_width = 800
frame_height = 600

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO("../YOLO-Weights/Speed_limit.pt")

class_dict = {0: 'Speed Limit 10', 1: 'Speed Limit 100', 2: 'Speed Limit 110', 3: 'Speed Limit 120', 4: 'Speed Limit 20', 5: 'Speed Limit 30', 6: 'Speed Limit 40', 7: 'Speed Limit 50', 8: 'Speed Limit 60', 9: 'Speed Limit 70', 
10: 'Speed Limit 80', 11: 'Speed Limit 90', 12: 'Stop'}

# Create a window and set its size
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", frame_width, frame_height)

while True:
    success, img = cap.read()

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

            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    out.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

out.release()
