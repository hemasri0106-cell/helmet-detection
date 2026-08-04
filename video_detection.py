import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture("in_traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame) # reset to default confidence threshold
    annotated_frame = results[0].plot()

    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()