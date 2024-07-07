import cv2
import torch
import pytesseract
from datetime import datetime
import os
import subprocess
import time
from fuzzywuzzy import fuzz, process

# 사용자 딕셔너리 정의
user_duldul = {
    '이민진': 'A',
    '정다운': 'B',
    '이재진': 'C',
    '제갈준영': 'D'
}

# Tesseract 실행 파일 경로 설정 (필요한 경우)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Tesseract 설치 경로

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/invoice15/weights/best.pt')
model.eval()

# 결과 저장 경로 설정
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# duldul_list 초기화
duldul_list = []

# 캡처된 송장 수
capture_count = 0
max_captures = 3

def find_best_match(extracted_text, user_duldul):
    best_match = process.extractOne(extracted_text, user_duldul.keys(), scorer=fuzz.token_sort_ratio)
    return best_match

while capture_count < max_captures:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # YOLOv5 모델을 사용하여 객체 검출
    results = model(frame)

    # 현재 시간을 기반으로 타임스탬프 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 결과에서 바운딩 박스 추출
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'

        # 'invoice' 클래스이고, 정확도가 0.85 이상일 때
        if model.names[int(cls)] == 'invoice' and conf >= 0.85:
            bbox_area = (x2 - x1) * (y2 - y1)
            img_area = frame.shape[0] * frame.shape[1]

            # 바운딩 박스가 전체 이미지의 50% 이상일 때
            if bbox_area >= 0.5 * img_area:
                # tts/speak.py 실행 (이미지 캡처 시)
                # subprocess.run(['python', 'tts/speak.py'])

                # 이미지 캡처
                captured_image_path = os.path.join(output_dir, f'invoice_{timestamp}.png')
                cv2.imwrite(captured_image_path, frame[y1:y2, x1:x2])

                # Tesseract OCR로 텍스트 추출 (한글 데이터 사용)
                text = pytesseract.image_to_string(frame[y1:y2, x1:x2], lang='kor')
                if text.strip():  # 추출된 텍스트가 있는지 확인
                    best_match, score = find_best_match(text, user_duldul)
                    if score >= 80:  # 임계값 설정
                        matched_value = user_duldul[best_match]
                        if not duldul_list or duldul_list[-1] != matched_value:
                            duldul_list.append(matched_value)
                            print(f"Matched: {text} -> {best_match} (score: {score})")
                            print(f"Captured Image Path: {captured_image_path}")
                            print("Extracted Text:")
                            print(text)
                            # 캡처된 송장 수 증가
                            capture_count += 1
                            print(f"OCR 완료되었습니다. 현재 캡처된 송장 수: {capture_count}")

                            # 5초 대기
                            time.sleep(5)

                            # 3개의 송장이 캡처되면 루프 종료
                            if capture_count >= max_captures:
                                break

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 결과 표시
    cv2.imshow('YOLOv5 Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 윈도우 닫기
cap.release()
cv2.destroyAllWindows()

# 알파벳 순으로 정렬
duldul_shiplist = sorted(duldul_list)

# OCR 텍스트 파일로 저장
if duldul_shiplist:
    ocr_output_path = os.path.join(output_dir, f'ocr_results_{timestamp}.txt')
    with open(ocr_output_path, 'w') as f:
        for idx, text in enumerate(duldul_shiplist, start=1):
            f.write(f"Index {idx}:\n")
            f.write("Extracted Text:\n")
            f.write(text)
            f.write("\n\n")
    print(f"OCR results saved to {ocr_output_path}")
else:
    print("No OCR results to save.")

# 최종 duldul_shiplist 출력
print("최종 duldul_shiplist:", duldul_shiplist)
