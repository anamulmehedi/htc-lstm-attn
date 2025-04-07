import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

def preprocess_data(file_path, time_steps=12):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Exclude flat stations
    flat_stations = [10320, 11316, 41858, 41909, 41926, 41958, 41977, 11925, 11929, 12103, 12110]
    data = data[~data['Station Code'].isin(flat_stations)]

    # Preserve Date for seasonal features
    data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(Day=1))
    data.set_index('Date', inplace=True)

    # Drop non-essential columns
    data.drop(columns=['Area Type'], inplace=True, errors='ignore')

    # Handle missing values using KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    numerical_cols = ['Solar Radiation', 'ETo', 'PET', 'Sunshine (Hours)', 'Wind Speed (m/s)',
                      'Cloud Coverage (Octs)', 'Humidity (percent)', 'Total Rainfall (mm)', 'Temperature (Deg.Cel)']
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    # Add seasonal and temporal features
    data['Month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data.index.month / 12)

    # Add lagged features and rolling statistics
    target_variables = ['Temperature (Deg.Cel)', 'Humidity (percent)']
    for target in target_variables:
        for lag in [1, 2, 3]:
            data[f'{target}_lag{lag}'] = data[target].shift(lag)
        data[f'{target}_rolling_mean_3'] = data[target].rolling(window=3).mean()
        data[f'{target}_rolling_mean_7'] = data[target].rolling(window=7).mean()
        data[f'{target}_rolling_std_3'] = data[target].rolling(window=3).std()

    data.dropna(inplace=True)

    # Normalize features
    scaler = StandardScaler()
    features = data.drop(columns=target_variables + ['Station Code']).columns
    data[features] = scaler.fit_transform(data[features])

    # Normalize targets
    target_scaler_temp = MinMaxScaler(feature_range=(0.2, 0.8))
    target_scaler_hum = MinMaxScaler(feature_range=(0.2, 0.8))
    data['Temperature (Deg.Cel)'] = target_scaler_temp.fit_transform(data[['Temperature (Deg.Cel)']])
    data['Humidity (percent)'] = target_scaler_hum.fit_transform(data[['Humidity (percent)']])

    return data, scaler, target_scaler_temp, target_scaler_hum

def create_sequences(X, y_temp, y_hum, time_steps=12):
    X_seq, y_seq_temp, y_seq_hum = [], [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X.iloc[i:i + time_steps].values)
        y_seq_temp.append(y_temp.iloc[i + time_steps])
        y_seq_hum.append(y_hum.iloc[i + time_steps])
    return np.array(X_seq), np.array(y_seq_temp), np.array(y_seq_hum)
