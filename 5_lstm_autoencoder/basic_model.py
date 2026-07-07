import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# 모델 정의 (이전 아티팩트에서 정의한 것과 동일)
def create_lstm_ae_model(input_shape, latent_dim=32):
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)
    latent_space = Dense(latent_dim)(encoded)
    decoded = RepeatVector(input_shape[0])(latent_space)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    return Model(inputs, outputs)

# 예시 데이터 생성
def generate_trajectory(n_samples, seq_length, n_features, anomaly=False):
    trajectory = np.zeros((n_samples, seq_length, n_features))
    for i in range(n_samples):
        # 기본 궤적: 직선 운동
        x = np.linspace(0, 10, seq_length)
        y = np.linspace(0, 10, seq_length)
        
        if anomaly:
            # 이상 궤적: 중간에 갑자기 방향 전환
            mid = seq_length // 2
            y[mid:] = 10 - np.linspace(0, 5, seq_length - mid)
        
        # 속도, 가속도 계산 (간단한 예시)
        vx = np.gradient(x)
        vy = np.gradient(y)
        ax = np.gradient(vx)
        ay = np.gradient(vy)
        
        # 각도와 각속도 (간단한 예시)
        theta = np.arctan2(vy, vx)
        omega = np.gradient(theta)
        
        trajectory[i] = np.column_stack((x, y, vx, vy, ax, ay, theta, omega))
    
    return trajectory

# 파라미터 설정
seq_length = 100
n_features = 8
n_samples = 1000

# 데이터 생성
normal_data = generate_trajectory(n_samples, seq_length, n_features)
anomaly_data = generate_trajectory(n_samples // 10, seq_length, n_features, anomaly=True)

# 모델 생성 및 학습
model = create_lstm_ae_model((seq_length, n_features))
model.compile(optimizer='adam', loss='mse')
model.fit(normal_data, normal_data, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# 정상 및 이상 데이터에 대한 재구성 오차 계산
normal_reconstructed = model.predict(normal_data)
anomaly_reconstructed = model.predict(anomaly_data)

normal_mse = np.mean(np.square(normal_data - normal_reconstructed), axis=(1,2))
anomaly_mse = np.mean(np.square(anomaly_data - anomaly_reconstructed), axis=(1,2))

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.hist(normal_mse, bins=50, alpha=0.5, label='Normal')
plt.hist(anomaly_mse, bins=50, alpha=0.5, label='Anomaly')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Reconstruction Errors')
plt.show()

# 임계값 설정 및 이상 탐지
threshold = np.percentile(normal_mse, 95)  # 95th percentile of normal errors
print(f"Threshold: {threshold}")

detected_anomalies = anomaly_mse > threshold
print(f"Detected {np.sum(detected_anomalies)} out of {len(anomaly_mse)} anomalies")

# 예시 궤적 시각화
def plot_trajectory(traj, title):
    plt.figure(figsize=(8, 8))
    plt.plot(traj[:, 0], traj[:, 1])
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

plot_trajectory(normal_data[0], "Normal Trajectory")
plot_trajectory(anomaly_data[0], "Anomaly Trajectory")