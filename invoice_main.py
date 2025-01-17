import cv2
import torch
import pytesseract
from datetime import datetime
import os
import time
import pandas as pd
from process_duldul_list import process_duldul_list  # 함수 불러오기
from fuzzywuzzy import fuzz

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

# CSV 파일 읽기 (cp949 인코딩 사용)
info_df = pd.read_csv('info.csv', encoding='cp949')
user_df = pd.read_csv('user.csv', encoding='cp949')

# duldul_list 초기화
duldul_list = info_df.copy()
duldul_list['name'] = ''
duldul_list['location'] = ''
duldul_list['point_x'] = 0
duldul_list['point_y'] = 0
duldul_list['radian_lotation'] = 0

previous_names = []  # 이전에 캡처된 이름들을 저장하기 위한 리스트

def extract_name(text):
    # 정규 표현식이나 특정 패턴을 사용하여 이름 추출
    lines = text.split('\n')
    for line in lines:
        if line.strip():  # 비어있지 않은 라인
            return line.strip()
    return None

while True:
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

        # 'invoice' 클래스이고, 정확도가 0.80 이상일 때
        if model.names[int(cls)] == 'invoice' and conf >= 0.80:
            bbox_area = (x2 - x1) * (y2 - y1)
            img_area = frame.shape[0] * frame.shape[1]

            # 바운딩 박스가 전체 이미지의 40% 이상일 때
            if bbox_area >= 0.4 * img_area:
                # 이미지 캡처
                captured_image_path = os.path.join(output_dir, f'invoice_{timestamp}.png')
                cv2.imwrite(captured_image_path, frame[y1:y2, x1:x2])

                # Tesseract OCR로 텍스트 추출 (한글 데이터 사용)
                text = pytesseract.image_to_string(frame[y1:y2, x1:x2], lang='kor')
                name = extract_name(text)
                if name and all(fuzz.token_sort_ratio(name, prev_name) < 80 for prev_name in previous_names):
                    # 가장 비슷한 이름 찾기
                    best_match = None
                    best_score = 0
                    for _, row in user_df.iterrows():
                        score = fuzz.token_sort_ratio(name, row['name'])
                        if score > best_score:
                            best_match = row
                            best_score = score
                    
                    if best_match is not None and best_match['name'] not in previous_names:
                        new_row = info_df.iloc[0].copy()
                        new_row['name'] = best_match['name']
                        new_row['location'] = best_match['location']
                        new_row['point_x'] = best_match['point_x']
                        new_row['point_y'] = best_match['point_y']
                        new_row['radian_lotation'] = best_match['radian_lotation']
                        
                        duldul_list = duldul_list.append(new_row, ignore_index=True)
                        previous_names.append(best_match['name'])

                        print(f"Captured Image Path: {captured_image_path}")
                        print("Extracted Text:")
                        print(text)
                        print(f"Matched Name: {best_match['name']}")

                        print(f"OCR 완료되었습니다. 현재 캡처된 송장 수: {len(duldul_list)}")

                        # 3초 대기
                        time.sleep(3)

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 결과 표시
    cv2.imshow('YOLOv5 Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # duldul_list를 입력으로 받아서 duldul_shiplist를 생성
    duldul_shiplist = process_duldul_list(duldul_list)

    # 서로 다른 알파벳 3개가 들어올 때까지 반복
    if len(duldul_shiplist) >= 3:
        break

# 웹캠 해제 및 윈도우 닫기
cap.release()
cv2.destroyAllWindows()

# OCR 텍스트 파일로 저장
if not duldul_list.empty:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 타임스탬프 다시 생성
    ocr_output_path = os.path.join(output_dir, f'ocr_results_{timestamp}.txt')
    with open(ocr_output_path, 'w', encoding='utf-8-sig') as f:
        for idx, row in duldul_list.iterrows():
            f.write(f"Index {idx+1}:\n")
            f.write("Extracted Text:\n")
            f.write(row['name'])
            f.write("\n\n")
    print(f"OCR results saved to {ocr_output_path}")
else:
    print("No OCR results to save.")

# 최종 duldul_list 출력
print("최종 duldul_list:", duldul_list)

# 최종 duldul_shiplist 알파벳 순 정렬 및 출력
duldul_shiplist = duldul_list.sort_values(by='location').reset_index(drop=True)
print("최종 duldul_shiplist:", duldul_shiplist)

# duldul_shiplist를 CSV로 저장 (utf-8-sig 인코딩 사용)
duldul_shiplist.to_csv(os.path.join(output_dir, 'duldul_shiplist.csv'), index=False, encoding='utf-8-sig')
print("duldul_shiplist saved to duldul_shiplist.csv")
