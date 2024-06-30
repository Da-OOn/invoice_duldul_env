import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/invoice14/weights/best.pt')
model.eval()

# 웹캠 설정
cap = cv2.VideoCapture(0)  # 0번 카메라 (기본 웹캠)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # YOLOv5 모델을 사용하여 객체 검출
    results = model(frame)

    # 결과에서 바운딩 박스 추출
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 영상 출력
    cv2.imshow('YOLOv5 Real-time Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()