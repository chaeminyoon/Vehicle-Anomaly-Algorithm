# LSTM 오토인코더를 활용한 차량 이상궤적 식별 정확도 향상에 관한 연구

CCTV 영상에서 차량 궤적을 추출하고, LSTM 오토인코더(LSTM Autoencoder)로 정상 궤적 패턴을 학습하여 **이상궤적(비정상 주행 패턴)을 식별**하는 석사학위논문 연구 코드입니다.

## 연구 개요

- 전국 40개 지점(지점 11~50)의 도로 CCTV 영상에서 YOLOv8 기반 객체 탐지·추적으로 차량 궤적 데이터를 수집
- 차량 궤적의 횡방향 밀도 히스토그램 피크 분석으로 도로 정보(차로 수, 차선 위치, 곡률)를 추출 (DeepLabV3/CLRNet 기반 영상 차선 검출도 병행 실험)
- 지점별 도로 특성(차선 수, 평균속도, 통행량)을 클러스터링(K-means, DBSCAN, GMM, 계층적 군집)하여 유사 지점을 그룹화
- 정상 궤적 시퀀스로 LSTM 오토인코더를 학습하고, 재구성 오차(reconstruction error) 기반으로 이상궤적을 식별
- 클러스터별 모델과 전체 통합 모델의 이상궤적 식별 정확도를 비교하여 정확도 향상을 검증

## 파이프라인

```
CCTV 영상 (지점 11~50)
   │
   ▼
[1] 궤적 추출 ──── YOLOv8 + 객체 추적 → TrackID별 (X, Y, Time) CSV
   │
   ▼
[2] 차선 검출 ──── 차로 수·도로 곡률 추정 → 도로 정보 CSV
   │
   ▼
[3] 데이터 전처리 ─ 짧은 궤적 제거, 상대좌표/속도 변환, 시퀀스 생성(길이 50)
   │
   ▼
[4] 지점 클러스터링 ─ K-means / DBSCAN / GMM / 계층적 군집 비교
   │
   ▼
[5] LSTM 오토인코더 ─ 정상 궤적 학습 → 재구성 오차로 이상궤적 판별
   │
   ▼
[6] 평가 ────────── 이상궤적 테스트 데이터 생성, 모델별 정확도 비교
```

## 디렉터리 구성

| 디렉터리 | 내용 |
|---|---|
| `1_trajectory_extraction/` | YOLOv8 기반 차량 탐지·추적 및 궤적 CSV 생성, 궤적 시각화 검증 |
| `2_lane_detection/` | 차선 검출, 차로 수 카운트, 도로 곡률 추정, 차선 라벨링 도구 |
| `3_preprocessing/` | 궤적 데이터 전처리(시퀀스화, 좌표 변환), 도로 정보 결합 |
| `4_clustering/` | 지점별 도로 특성 클러스터링 (K-means, DBSCAN, GMM, 계층적) 및 클러스터별 모델 생성 |
| `5_lstm_autoencoder/` | LSTM 오토인코더 모델 정의·학습 (모듈화 버전 + 통합 학습 스크립트) |
| `6_evaluation/` | 이상궤적 테스트 데이터 생성기(pygame), 정확도 비교, 평균속도·차량수 검증 |
| `docs/images/` | 결과 예시 이미지 |

## 주요 코드

### 1. 궤적 추출 (`1_trajectory_extraction/`)
- `trajectory_yolo8.py` — YOLOv8(car/bus/truck 클래스)로 다중 영상을 멀티프로세싱 처리, TrackID별 중심좌표 시계열을 CSV로 저장
- `track_validation.py` — 추출된 궤적 CSV를 시각화하여 추적 품질 검증

### 2. 차선 검출 (`2_lane_detection/`)
- `lane_detection.py`, `lane_count.py` — 차량 궤적의 X좌표 히스토그램을 가우시안 스무딩 후 피크 검출(`find_peaks`)하여 차선 수·차선 위치 추정
- `curve_estimator.py` — 도로 곡률 추정
- `line_labelling.py` — 차선 수동 라벨링 도구

### 3. 전처리 (`3_preprocessing/`)
- `preprocess.py` — 짧은 궤적 제거, 상대좌표/속도 기반 변환, 고정 길이 시퀀스 생성
- `trajectory_preprocessing.py`, `road_info.py` — 지점별 궤적 통합 및 도로 정보 결합

### 4. 클러스터링 (`4_clustering/`)
- 지점별 특성(차선 수, 평균속도, 통행량)으로 유사 지점 그룹화
- 실루엣 계수 등으로 K-means / DBSCAN / GMM / 계층적 군집 비교
- `kmeans_model.py`, `gmm_model.py` — 클러스터별 LSTM 오토인코더 모델 생성

### 5. LSTM 오토인코더 (`5_lstm_autoencoder/`)
- `model.py` — LSTM Encoder → Bottleneck(Dense) → RepeatVector → LSTM Decoder 구조 (TensorFlow/Keras)
- `dataset.py` — 속도·가속도 특징 생성, MinMax 정규화, 시퀀스 패딩
- `train_full_model.py` — 전체 지점 통합 모델 학습 (시퀀스 길이 50, 슬라이딩 윈도우)

### 6. 평가 (`6_evaluation/`)
- `anomaly_trajectory_generator.py` — 영상 위에 이상궤적을 직접 그려 테스트 데이터를 생성하는 pygame 도구
- `make_test_data.py` — 이상궤적 TrackID 필터링으로 테스트셋 구성
- `accuracy_comparison.py` — 추정 결과의 정확도(MAE, RMSE, ±1 일치율) 비교

## 실행 검증 및 결과 분석

전 단계 코드를 실제 데이터로 실행 검증한 결과와 분석은 **[docs/ANALYSIS.md](docs/ANALYSIS.md)** 참고.
(YOLOv8 추적, 히스토그램 차선 추정, LSTM 오토인코더 학습·이상궤적 식별까지 전 파이프라인 동작 확인)

## 결과 예시

| 차량 궤적 추출 | 차선 검출 |
|---|---|
| ![trajectories](docs/images/trajectories_location_11.png) | ![lanes](docs/images/detected_lanes_14.png) |

| 통행 시간 분포 | 궤적 시각화 |
|---|---|
| ![elapsed](docs/images/elapsed_time_distribution_location_11.png) | ![visualization](docs/images/visualization_11.png) |

## 실행 환경

```bash
pip install -r requirements.txt
```

- Python 3.9+
- 주요 라이브러리: ultralytics(YOLOv8), OpenCV, TensorFlow/Keras, scikit-learn, pandas

> **참고:** CCTV 원본 영상, 궤적 CSV 원본 데이터, 학습된 모델 가중치(`.pt`, `.pth`)는 용량 및 데이터 제공 조건상 저장소에 포함하지 않았습니다. 스크립트 내 데이터 경로는 로컬 환경에 맞게 수정이 필요합니다.

## 논문 정보

- **제목:** LSTM 오토인코더를 활용한 차량이상궤적식별 정확도 향상에 관한 연구
- **저자:** 윤채민 (석사학위논문)
