import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# 모델 로드
model = load_model(r"C:\Users\kimyoungjin\cow_marbling_resnet_model.h5")  # h5 폴더 경로 (수정 필요)

# 카메라 열기
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("카메라가 열렸습니다. 'q'를 눌러 종료하세요.")

# 타이머 초기화
last_prediction_time = 0
interval = 0.1  # 0.1초 간격으로 예측

while True:
    ret, frame = camera.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    processed_frame = frame.copy()

    # 밝은 영역 강조를 위한 밝기(Lightness) 추출
    brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]  # HSV 색상공간에서 V(밝기) 채널 추출

    # 밝은 영역 강조 (Threshold)
    _, thresh = cv2.threshold(brightness, 200, 255, cv2.THRESH_BINARY)

    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽선만 선택
    largest_contour = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 노이즈 제거: 500 픽셀 이하 영역 제외
            if area > largest_area:  # 가장 큰 윤곽선 선택
                largest_area = area
                largest_contour = contour

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 빨간색 사각형

        roi = frame[y:y + h, x:x + w]  # ROI 추출
        current_time = time.time()

        if current_time - last_prediction_time >= interval:
            last_prediction_time = current_time

            # ROI 전처리
            resized_roi = cv2.resize(roi, (128, 128))
            roi_array = np.expand_dims(img_to_array(resized_roi) / 255.0, axis=0)

            # 모델 예측
            predictions = model.predict(roi_array)
            class_idx = np.argmax(predictions[0])

            # 등급 출력
            if class_idx == 2:
                grade = '3등급'
            elif class_idx == 0:
                grade = '1등급'
            else:
                grade = '1++등급'

            print(f"Predicted class: {class_idx}")
            print(f"Predicted probabilities: {predictions}")
            print(f"등급: {grade}")

            # 결과를 ROI 위에 표시
            cv2.putText(processed_frame, grade, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 결과 화면 표시
    cv2.imshow("Marbling Detection", processed_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
camera.release()
cv2.destroyAllWindows()
