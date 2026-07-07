# config.py

class Config:
    # 데이터 관련 설정
    DATA_PATH = 'path/to/trajectory_data.csv'
    SEQUENCE_LENGTH = 50
    TRAIN_SPLIT = 0.8

    # 모델 관련 설정
    ENCODING_DIM = 16
    LSTM_UNITS = [64, 32]

    # 학습 관련 설정
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # 이상 탐지 관련 설정
    ANOMALY_THRESHOLD_PERCENTILE = 95
