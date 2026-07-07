import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_paths):
    all_data = []
    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)
        df['CCTV_ID'] = i  # Add CCTV ID
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def prepare_sequences(df, seq_length=50):
    sequences = []
    for cctv_id in df['CCTV_ID'].unique():
        cctv_data = df[df['CCTV_ID'] == cctv_id]
        for track_id in cctv_data['TrackID'].unique():
            track_data = cctv_data[cctv_data['TrackID'] == track_id].sort_values('Time')
            coords = track_data[['X_center', 'Y_center']].values
            if len(coords) >= seq_length:
                for i in range(len(coords) - seq_length + 1):
                    sequences.append(coords[i:i+seq_length])
    return np.array(sequences)

def build_lstm_autoencoder(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
        tf.keras.layers.RepeatVector(input_shape[0]),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_base_model(sequences, epochs=50, batch_size=32):
    train_data, val_data = train_test_split(sequences, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    val_data_scaled = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(val_data.shape)
    
    model = build_lstm_autoencoder((train_data.shape[1], train_data.shape[2]))
    
    history = model.fit(
        train_data_scaled, train_data_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data_scaled, val_data_scaled),
        shuffle=True
    )
    
    return model, scaler, history

def fine_tune_model(base_model, cctv_data, scaler, epochs=10, batch_size=32):
    cctv_data_scaled = scaler.transform(cctv_data.reshape(-1, cctv_data.shape[-1])).reshape(cctv_data.shape)
    
    fine_tuned_model = tf.keras.models.clone_model(base_model)
    fine_tuned_model.set_weights(base_model.get_weights())
    fine_tuned_model.compile(optimizer='adam', loss='mse')
    
    history = fine_tuned_model.fit(
        cctv_data_scaled, cctv_data_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True
    )
    
    return fine_tuned_model, history

def detect_anomalies(model, data, scaler, threshold_percentile=95):
    data_scaled = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    reconstructions = model.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=(1,2))
    threshold = np.percentile(mse, threshold_percentile)
    anomalies = mse > threshold
    return anomalies, mse

# 메인 실행 코드
file_paths = [f'D:/trajectory_project/trajectory_csv/track_history_{i}.csv' for i in range(11,51)]  # 실제 파일 경로로 수정 필요

combined_df = load_and_preprocess_data(file_paths)
sequences = prepare_sequences(combined_df)

# 기본 모델 학습
base_model, scaler, base_history = train_base_model(sequences)

# CCTV별 fine-tuning 및 이상 탐지
for cctv_id in combined_df['CCTV_ID'].unique():
    cctv_data = combined_df[combined_df['CCTV_ID'] == cctv_id]
    cctv_sequences = prepare_sequences(cctv_data)
    
    fine_tuned_model, ft_history = fine_tune_model(base_model, cctv_sequences, scaler)
    
    anomalies, mse = detect_anomalies(fine_tuned_model, cctv_sequences, scaler)
    
    print(f"CCTV {cctv_id}: {np.sum(anomalies)} anomalies detected out of {len(anomalies)} sequences")

    # 결과 시각화 (예시)
    plt.figure(figsize=(12, 6))
    plt.plot(mse)
    plt.axhline(y=np.percentile(mse, 95), color='r', linestyle='--')
    plt.title(f'Reconstruction Error for CCTV {cctv_id}')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Sequence Index')
    plt.show()