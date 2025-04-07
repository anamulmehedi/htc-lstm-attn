import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import create_sequences
import os

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(model, test_data, data, target_scaler_temp, target_scaler_hum, save_dir):
    X_test_seq, y_test_seq_temp, y_test_seq_hum = test_data

    # Evaluate overall data
    y_pred_temp, y_pred_hum = model.predict(X_test_seq)
    y_pred_temp_unscaled = target_scaler_temp.inverse_transform(y_pred_temp)
    y_pred_hum_unscaled = target_scaler_hum.inverse_transform(y_pred_hum)
    y_test_temp_unscaled = target_scaler_temp.inverse_transform(y_test_seq_temp.reshape(-1, 1))
    y_test_hum_unscaled = target_scaler_hum.inverse_transform(y_test_seq_hum.reshape(-1, 1))

    mae_temp = mean_absolute_error(y_test_temp_unscaled, y_pred_temp_unscaled)
    rmse_temp = np.sqrt(mean_squared_error(y_test_temp_unscaled, y_pred_temp_unscaled))
    mape_temp = mean_absolute_percentage_error(y_test_temp_unscaled, y_pred_temp_unscaled)
    r2_temp = r2_score(y_test_temp_unscaled, y_pred_temp_unscaled)

    mae_hum = mean_absolute_error(y_test_hum_unscaled, y_pred_hum_unscaled)
    rmse_hum = np.sqrt(mean_squared_error(y_test_hum_unscaled, y_pred_hum_unscaled))
    mape_hum = mean_absolute_percentage_error(y_test_hum_unscaled, y_pred_hum_unscaled)
    r2_hum = r2_score(y_test_hum_unscaled, y_pred_hum_unscaled)

    print("\n=== Overall Evaluation Metrics for HTC-LSTM-Attn (All Valid Stations) ===")
    print(f"Temperature Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae_temp:.4f} °C")
    print(f"  - Root Mean Squared Error (RMSE): {rmse_temp:.4f} °C")
    print(f"  - Mean Absolute Percentage Error (MAPE): {mape_temp:.4f} %")
    print(f"  - R-squared (R²): {r2_temp:.4f}")
    print(f"Humidity Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae_hum:.4f} %")
    print(f"  - Root Mean Squared Error (RMSE): {rmse_hum:.4f} %")
    print(f"  - Mean Absolute Percentage Error (MAPE): {mape_hum:.4f} %")
    print(f"  - R-squared (R²): {r2_hum:.4f}")

    # Save overall predictions and actuals
    np.save(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_pred_temp.npy'), y_pred_temp_unscaled)
    np.save(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_pred_hum.npy'), y_pred_hum_unscaled)
    np.save(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_actual_temp.npy'), y_test_temp_unscaled)
    np.save(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_actual_hum.npy'), y_test_hum_unscaled)

    # Save overall metrics
    metrics_df = pd.DataFrame({
        'Temp_MAE': [mae_temp], 'Temp_RMSE': [rmse_temp], 'Temp_MAPE': [mape_temp], 'Temp_R2': [r2_temp],
        'Hum_MAE': [mae_hum], 'Hum_RMSE': [rmse_hum], 'Hum_MAPE': [mape_hum], 'Hum_R2': [r2_hum]
    })
    metrics_df['Model'] = 'HTC_LSTM_Attn'
    metrics_df.to_csv(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_metrics.csv'), index=False)

    # Evaluate each station
    stations = data['Station Code'].unique()
    station_metrics_list = []
    time_steps = 12

    for station in stations:
        station_data = data[data['Station Code'] == station]
        X_station = station_data.drop(columns=['Temperature (Deg.Cel)', 'Humidity (percent)', 'Station Code'])
        y_station_temp = station_data['Temperature (Deg.Cel)']
        y_station_hum = station_data['Humidity (percent)']
        X_station_seq, y_station_seq_temp, y_station_seq_hum = create_sequences(X_station, y_station_temp, y_station_hum, time_steps)

        if len(X_station_seq) > 0:
            station_pred_temp, station_pred_hum = model.predict(X_station_seq)
            station_pred_temp_unscaled = target_scaler_temp.inverse_transform(station_pred_temp)
            station_pred_hum_unscaled = target_scaler_hum.inverse_transform(station_pred_hum)
            y_station_temp_unscaled = target_scaler_temp.inverse_transform(y_station_seq_temp.reshape(-1, 1))
            y_station_hum_unscaled = target_scaler_hum.inverse_transform(y_station_seq_hum.reshape(-1, 1))

            mae_temp = mean_absolute_error(y_station_temp_unscaled, station_pred_temp_unscaled)
            rmse_temp = np.sqrt(mean_squared_error(y_station_temp_unscaled, station_pred_temp_unscaled))
            mape_temp = mean_absolute_percentage_error(y_station_temp_unscaled, station_pred_temp_unscaled)
            r2_temp = r2_score(y_station_temp_unscaled, station_pred_temp_unscaled)

            mae_hum = mean_absolute_error(y_station_hum_unscaled, station_pred_hum_unscaled)
            rmse_hum = np.sqrt(mean_squared_error(y_station_hum_unscaled, station_pred_hum_unscaled))
            mape_hum = mean_absolute_percentage_error(y_station_hum_unscaled, station_pred_hum_unscaled)
            r2_hum = r2_score(y_station_hum_unscaled, station_pred_hum_unscaled)

            print(f"\nStation {station} - Temperature: MAE = {mae_temp:.4f}, RMSE = {rmse_temp:.4f}, MAPE = {mape_temp:.4f}, R-squared = {r2_temp:.4f}")
            print(f"Station {station} - Humidity: MAE = {mae_hum:.4f}, RMSE = {rmse_hum:.4f}, MAPE = {mape_hum:.4f}, R-squared = {r2_hum:.4f}")

            station_metrics = {
                'Station': station,
                'Temp_MAE': mae_temp,
                'Temp_RMSE': rmse_temp,
                'Temp_MAPE': mape_temp,
                'Temp_R2': r2_temp,
                'Hum_MAE': mae_hum,
                'Hum_RMSE': rmse_hum,
                'Hum_MAPE': mape_hum,
                'Hum_R2': r2_hum
            }
            station_metrics_list.append(station_metrics)

            np.save(os.path.join(save_dir, f'HTC_LSTM_Attn_station_{station}_pred_temp.npy'), station_pred_temp_unscaled)
            np.save(os.path.join(save_dir, f'HTC_LSTM_Attn_station_{station}_pred_hum.npy'), station_pred_hum_unscaled)
            np.save(os.path.join(save_dir, f'HTC_LSTM_Attn_station_{station}_actual_temp.npy'), y_station_temp_unscaled)
            np.save(os.path.join(save_dir, f'HTC_LSTM_Attn_station_{station}_actual_hum.npy'), y_station_hum_unscaled)

    # Save station-level metrics
    station_metrics_df = pd.DataFrame(station_metrics_list)
    station_metrics_df.to_csv(os.path.join(save_dir, 'HTC_LSTM_Attn_station_metrics.csv'), index=False)

    return y_pred_temp_unscaled, y_pred_hum_unscaled, y_test_temp_unscaled, y_test_hum_unscaled, station_metrics_list

if __name__ == "__main__":
    import tensorflow as tf
    model = tf.keras.models.load_model("results/htc_lstm_attn/overall/best_model_htc_lstm_attn_overall.h5")
    data_path = "data/Organized_Weather_Data.csv"
    save_dir = "results"
    data, scaler, target_scaler_temp, target_scaler_hum = preprocess_data(data_path)
    X = data.drop(columns=['Temperature (Deg.Cel)', 'Humidity (percent)', 'Station Code'])
    y_temp = data['Temperature (Deg.Cel)']
    y_hum = data['Humidity (percent)']
    X_seq, y_seq_temp, y_seq_hum = create_sequences(X, y_temp, y_hum, 12)
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)
    test_data = (X_seq[train_size + val_size:], y_seq_temp[train_size + val_size:], y_seq_hum[train_size + val_size:])
    evaluate_model(model, test_data, data, target_scaler_temp, target_scaler_hum, save_dir)
