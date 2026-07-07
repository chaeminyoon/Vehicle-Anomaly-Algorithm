# main.py
import numpy as np
from config import Config
from dataset import TrajectoryDataset
from train import train_model, detect_anomalies
from utils import evaluate_model

def main():
    config = Config()
    
    # 데이터 로드 및 전처리
    dataset = TrajectoryDataset(config)
    sequences = dataset.load_data()
    train_data, test_data = dataset.split_data(sequences)
    
    # 모델 학습
    model, history = train_model(config, train_data)
    
    # 이상 탐지 임계값 설정
    train_reconstructions = model.predict(train_data)
    train_mse = np.mean(np.square(train_data - train_reconstructions), axis=(1,2))
    threshold = np.percentile(train_mse, config.ANOMALY_THRESHOLD_PERCENTILE)
    
    # 테스트 데이터에 대한 이상 탐지
    anomalies = detect_anomalies(model, test_data, threshold)
    
    # 성능 평가 (이 부분은 실제 라벨이 있다고 가정)
    # 실제 상황에서는 수동으로 라벨링된 데이터를 사용해야 합니다.
    np.random.seed(42)
    true_labels = np.random.randint(0, 2, size=len(test_data))
    evaluate_model(model, test_data, true_labels, threshold)

if __name__ == "__main__":
    main()
