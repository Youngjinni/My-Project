*개요*
이 프로젝트는 LSTM(Long Short-Trem Memory) 신경망을 활용하여 삼성전자(ticker : 005930.KS)의 종가를 예측하는 모델입니다.
해당 모델은 TensorFlow 및 Keras를 기반으로 구현했습니다.

datasets
출처 : Yahoo Finance
기간 : 2024년 1월 1일 이후(수정 가능)

구조
LSTM layer : 128unit, ReLU 활성화 함수 사용
Dense layer : 1
손실함수 : MAE(평균절댓값 오차)
최적화 : Adam
설정 : Epoch 200, batch 1
1. 데이터 다운로드 및 종가 전처리
2. train data 70% / test data 30%로 분할
3. LSTM 모델 학습 및 MAE 최소화
4. test data로 성능 평가 및 추가 지표 계산(MAPE)

결과 및 시각화
1. 주가 history
2. Epoch에 따른 MAE 변화
3. test data의 실제 종가와 예측 종가 비교
4. 일부 구간 확대 비교

24년 1월1일 - 25년 9월 1일 데이터 기준
MAE : 919.5222778320312
MAPE : 0.01515969
