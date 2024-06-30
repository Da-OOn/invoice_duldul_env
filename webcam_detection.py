import torch
import cv2
from PIL import Image
import numpy as np

class YOLOv5Detector:
    def __init__(self, model_path, conf_thres=0.25):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        # Convert the frame to an Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.model(img)
        results = results.pandas().xyxy[0]
        results = results[results['confidence'] >= self.conf_thres]
        return results

def draw_boxes(frame, results):
    for _, row in results.iterrows():
        x1, y1, x2, y2, conf, cls, name = row
        label = f"{name} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

def main():
    model_path = 'models/yolov5s.pt'
    detector = YOLOv5Detector(model_path)

    cap = cv2.VideoCapture(0)  # 0번 카메라 (웹캠) 사용

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)
        frame = draw_boxes(frame, results)

        cv2.imshow('YOLOv5 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
