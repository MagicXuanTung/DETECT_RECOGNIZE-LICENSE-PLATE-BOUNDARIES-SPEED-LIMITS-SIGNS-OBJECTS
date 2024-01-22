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

model = YOLO("../YOLO-Weights/traffic_signs.pt")

class_dict = {0: 'bus_stop', 1: 'do_not_enter', 2: 'do_not_stop', 3: 'do_not_turn_l', 4: 'do_not_turn_r', 5: 'do_not_u_turn', 6: 'enter_left_lane', 7: 'green_light', 8: 'left_right_lane', 9: 'no_parking', 10: 
'parking', 11: 'ped_crossing', 12: 'ped_zebra_cross', 13: 'railway_crossing', 14: 'red_light', 15: 'stop', 16: 't_intersection_l', 17: 'traffic_light', 18: 'u_turn', 19: 'warning', 20: 'yellow_light'}

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
