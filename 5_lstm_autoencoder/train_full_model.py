import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# 데이터 불러오기
data = pd.read_csv('/home/cmyoon/personal/0930/non_outlier_data.csv')

# 사용할 특성 선택
data = data[['StationID','TrackID','Time', 'X', 'Y']]

# 각 TrackID별 프레임 수 계산
track_lengths = data.groupby('TrackID').size()

print("평균 시퀀스 길이:", track_lengths.mean())
print("중앙값 시퀀스 길이:", track_lengths.median())
print("최대 시퀀스 길이:", track_lengths.max())

# 시간 순으로 정렬
data = data.sort_values(['StationID', 'Time'])

# 정규화
scaler = MinMaxScaler()
data[['X', 'Y']] = scaler.fit_transform(data[['X', 'Y']])

# 시퀀스 생성 함수
def create_sequences(data, seq_length, step_size):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, step_size):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

sequence_length = 50  # 실험할 시퀀스 길이
step_size = 10  # 슬라이딩 윈도우 스텝 사이즈

# 각 TrackID별로 시퀀스 생성
all_sequences = []
for station_id in data['StationID'].unique():
    station_data = data[data['StationID'] == station_id]
    for track_id in station_data['TrackID'].unique():
        track_data = station_data[station_data['TrackID'] == track_id]
        track_values = track_data[['X', 'Y']].values
        sequences = create_sequences(track_values, sequence_length, step_size)
        if len(sequences) > 0:
            all_sequences.extend(sequences)
all_sequences = np.array(all_sequences)

# 전체 데이터에 대한 모델 생성 및 학습
def create_lstm_ae(input_shape):
    model = Sequential([
        LSTM(64, 
             activation='tanh', 
             recurrent_activation='sigmoid',
             unroll=False,
             use_bias=True,
             unit_forget_bias=True,
             input_shape=(input_shape[1], input_shape[2]), 
             return_sequences=False),
        RepeatVector(input_shape[1]),
        LSTM(64, 
             activation='tanh', 
             recurrent_activation='sigmoid',
             unroll=False,
             use_bias=True,
             unit_forget_bias=True,
             return_sequences=True),
        TimeDistributed(Dense(input_shape[2]))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model_total = create_lstm_ae(all_sequences.shape)

# 모델 학습
model_total.fit(
    all_sequences, all_sequences,
    epochs=10,
    batch_size=32,
    shuffle=False,
    validation_split=0.1
)

# 모델 저장
model_total.save('/home/cmyoon/personal/0930/model_total.h5')
