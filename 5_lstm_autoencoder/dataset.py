# dataset.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TrajectoryDataset:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()

    def load_data(self):
        df = pd.read_csv(self.config.DATA_PATH)
        return self._preprocess_data(df)

    def _preprocess_data(self, df):
        # 특징 추출
        df['velocity_x'] = df.groupby('vehicle_id')['x'].diff() / 0.1
        df['velocity_y'] = df.groupby('vehicle_id')['y'].diff() / 0.1
        df['acceleration_x'] = df.groupby('vehicle_id')['velocity_x'].diff() / 0.1
        df['acceleration_y'] = df.groupby('vehicle_id')['velocity_y'].diff() / 0.1

        # 정규화
        columns_to_normalize = ['x', 'y', 'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y']
        df[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])

        # 시퀀스 생성
        sequences = self._create_sequences(df)
        
        return sequences

    def _create_sequences(self, df):
        sequences = []
        for vehicle_id in df['vehicle_id'].unique():
            vehicle_data = df[df['vehicle_id'] == vehicle_id][['x', 'y', 'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y']].values
            for i in range(len(vehicle_data) - self.config.SEQUENCE_LENGTH + 1):
                sequences.append(vehicle_data[i:i+self.config.SEQUENCE_LENGTH])
        return np.array(sequences)

    def split_data(self, sequences):
        train_size = int(self.config.TRAIN_SPLIT * len(sequences))
        train_data = sequences[:train_size]
        test_data = sequences[train_size:]
        return train_data, test_data
