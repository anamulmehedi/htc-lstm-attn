import matplotlib.pyplot as plt
import os

def plot_results(y_pred_temp_unscaled, y_pred_hum_unscaled, y_test_temp_unscaled, y_test_hum_unscaled, save_dir, station=None):
    prefix = f"HTC_LSTM_Attn_station_{station}" if station else "HTC_LSTM_Attn_overall"
    title_prefix = f"Station {station}" if station else "Overall"

    # Line Plot for Temperature
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_temp_unscaled[:400], label='Actual Temperature', alpha=0.5, color='blue')
    plt.plot(y_pred_temp_unscaled[:400], label='HTC-LSTM-Attn Predicted Temperature', linestyle='--', color='orange')
    plt.xlabel('Time Steps (Aggregated)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.title(f'{title_prefix} Actual vs Predicted Temperature (HTC-LSTM-Attn)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_temp_line.png'))
    plt.close()

    # Scatter Plot for Temperature
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_temp_unscaled[:400], y_pred_temp_unscaled[:400],
                label='HTC-LSTM-Attn Predicted vs Actual', alpha=0.5, color='orange')
    plt.plot([min(y_test_temp_unscaled[:400]), max(y_test_temp_unscaled[:400])],
             [min(y_test_temp_unscaled[:400]), max(y_test_temp_unscaled[:400])],
             'k--', label='Perfect Prediction')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.legend()
    plt.title(f'{title_prefix} Actual vs Predicted Temperature Scatter (HTC-LSTM-Attn)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_temp_scatter.png'))
    plt.close()

    # Line Plot for Humidity
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_hum_unscaled[:400], label='Actual Humidity', alpha=0.5, color='blue')
    plt.plot(y_pred_hum_unscaled[:400], label='HTC-LSTM-Attn Predicted Humidity', linestyle='--', color='orange')
    plt.xlabel('Time Steps (Aggregated)')
    plt.ylabel('Humidity (%)')
    plt.legend()
    plt.title(f'{title_prefix} Actual vs Predicted Humidity (HTC-LSTM-Attn)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_hum_line.png'))
    plt.close()

    # Scatter Plot for Humidity
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_hum_unscaled[:400], y_pred_hum_unscaled[:400],
                label='HTC-LSTM-Attn Predicted vs Actual', alpha=0.5, color='orange')
    plt.plot([min(y_test_hum_unscaled[:400]), max(y_test_hum_unscaled[:400])],
             [min(y_test_hum_unscaled[:400]), max(y_test_hum_unscaled[:400])],
             'k--', label='Perfect Prediction')
    plt.xlabel('Actual Humidity (%)')
    plt.ylabel('Predicted Humidity (%)')
    plt.legend()
    plt.title(f'{title_prefix} Actual vs Predicted Humidity Scatter (HTC-LSTM-Attn)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_hum_scatter.png'))
    plt.close()

if __name__ == "__main__":
    import numpy as np
    save_dir = "results"
    y_pred_temp = np.load(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_pred_temp.npy'))
    y_pred_hum = np.load(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_pred_hum.npy'))
    y_test_temp = np.load(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_actual_temp.npy'))
    y_test_hum = np.load(os.path.join(save_dir, 'HTC_LSTM_Attn_overall_actual_hum.npy'))
    plot_results(y_pred_temp, y_pred_hum, y_test_temp, y_test_hum, save_dir)
