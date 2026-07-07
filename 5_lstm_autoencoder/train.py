# train.py
import numpy as np
from model import LSTMAutoencoder

def train_model(config, train_data, val_data=None):
    model = LSTMAutoencoder(config)
    autoencoder = model.build_model(input_shape=(config.SEQUENCE_LENGTH, train_data.shape[2]))
    
    if val_data is not None:
        history = autoencoder.fit(
            train_data, train_data,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(val_data, val_data),
            shuffle=True
        )
    else:
        history = autoencoder.fit(
            train_data, train_data,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=0.1,
            shuffle=True
        )
    
    return autoencoder, history

def detect_anomalies(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=(1,2))
    return mse > threshold
