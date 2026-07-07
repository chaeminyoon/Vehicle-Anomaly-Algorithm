# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

class LSTMAutoencoder:
    def __init__(self, config):
        self.config = config

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # Encoder
        encoded = inputs
        for units in self.config.LSTM_UNITS:
            encoded = LSTM(units, activation='relu', return_sequences=True)(encoded)
        encoded = LSTM(self.config.LSTM_UNITS[-1], activation='relu')(encoded)
        encoded = Dense(self.config.ENCODING_DIM, activation='relu')(encoded)
        
        # Decoder
        decoded = RepeatVector(input_shape[0])(encoded)
        for units in reversed(self.config.LSTM_UNITS):
            decoded = LSTM(units, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(input_shape[1]))(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
