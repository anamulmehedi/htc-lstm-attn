import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from models.htc_lstm_attn import build_htc_lstm_attn_model
from preprocessing import preprocess_data, create_sequences
import os

def train_model(data_path, save_dir, time_steps=12):
    # Preprocess data
    data, scaler, target_scaler_temp, target_scaler_hum = preprocess_data(data_path, time_steps)

    # Overall data sequences
    X = data.drop(columns=['Temperature (Deg.Cel)', 'Humidity (percent)', 'Station Code'])
    y_temp = data['Temperature (Deg.Cel)']
    y_hum = data['Humidity (percent)']
    X_seq, y_seq_temp, y_seq_hum = create_sequences(X, y_temp, y_hum, time_steps)

    # Chronological split
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)
    X_train_seq, y_train_seq_temp, y_train_seq_hum = X_seq[:train_size], y_seq_temp[:train_size], y_seq_hum[:train_size]
    X_val_seq, y_val_seq_temp, y_val_seq_hum = X_seq[train_size:train_size + val_size], y_seq_temp[train_size:train_size + val_size], y_seq_hum[train_size:train_size + val_size]
    X_test_seq, y_test_seq_temp, y_test_seq_hum = X_seq[train_size + val_size:], y_seq_temp[train_size + val_size:], y_seq_hum[train_size + val_size:]

    # Hyperparameter tuning
    def build_model(hp):
        hp_dict = {
            'filters1': hp.Int('filters1', 32, 128, step=32),
            'filters3': hp.Int('filters3', 32, 128, step=32),
            'filters5': hp.Int('filters5', 32, 128, step=32),
            'filters_combine': hp.Int('filters_combine', 64, 256, step=64),
            'lstm_units': hp.Int('lstm_units', 32, 128, step=32),
            'dense_units': hp.Int('dense_units', 64, 256, step=64),
            'dropout1': hp.Float('dropout1', 0.1, 0.5, step=0.1),
            'dropout2': hp.Float('dropout2', 0.1, 0.5, step=0.1),
            'dropout3': hp.Float('dropout3', 0.1, 0.5, step=0.1)
        }
        return build_htc_lstm_attn_model((time_steps, X_train_seq.shape[2]), hp_dict)

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=2,
        directory=os.path.join(save_dir, 'tuning_results'),
        project_name='htc_lstm_attn_station_specific'
    )

    tuner.search(X_train_seq, [y_train_seq_temp, y_train_seq_hum],
                 validation_data=(X_val_seq, [y_val_seq_temp, y_val_seq_hum]),
                 epochs=50,
                 batch_size=32,
                 callbacks=[EarlyStopping(patience=10)])

    # Creating best model by getting best hyperparameters values
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0].values
    best_model = build_htc_lstm_attn_model((time_steps, X_train_seq.shape[2]), best_hps)

    # Trainining: The best model for overall data
    best_model.fit(X_train_seq, [y_train_seq_temp, y_train_seq_hum],
                   validation_data=(X_val_seq, [y_val_seq_temp, y_val_seq_hum]),
                   epochs=50,
                   batch_size=32,
                   callbacks=[EarlyStopping(patience=10)],
                   verbose=1)

    # Save: The best model for overall data
    overall_save_dir = os.path.join(save_dir, 'htc_lstm_attn', 'overall')
    os.makedirs(overall_save_dir, exist_ok=True)
    best_model.save(os.path.join(overall_save_dir, 'best_model_htc_lstm_attn_overall.h5'))

    # Train and save models for each station
    stations = data['Station Code'].unique()
    for station in stations:
        station_data = data[data['Station Code'] == station]
        X_station = station_data.drop(columns=['Temperature (Deg.Cel)', 'Humidity (percent)', 'Station Code'])
        y_station_temp = station_data['Temperature (Deg.Cel)']
        y_station_hum = station_data['Humidity (percent)']
        X_station_seq, y_station_seq_temp, y_station_seq_hum = create_sequences(X_station, y_station_temp, y_station_hum, time_steps)

        if len(X_station_seq) > 0:
            station_save_dir = os.path.join(save_dir, 'htc_lstm_attn', f'station_{station}')
            os.makedirs(station_save_dir, exist_ok=True)
            best_model.save(os.path.join(station_save_dir, f'best_model_htc_lstm_attn_station_{station}.h5'))

    return best_model, best_hps, (X_train_seq, y_train_seq_temp, y_train_seq_hum), (X_val_seq, y_val_seq_temp, y_val_seq_hum), (X_test_seq, y_test_seq_temp, y_test_seq_hum), data, scaler, target_scaler_temp, target_scaler_hum

if __name__ == "__main__":
    data_path = "data/Organized_Weather_Data.csv"
    save_dir = "results"
    best_model, best_hps, train_data, val_data, test_data, data, scaler, target_scaler_temp, target_scaler_hum = train_model(data_path, save_dir)
