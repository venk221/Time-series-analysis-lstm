# extract_feature.py

import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_pattern):
    """
    Load and process CSV files matching the given file pattern, one at a time.
    """
    all_files = glob.glob(file_pattern)
    processed_data = []

    for filename in all_files:
        print("Processing file: ", filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        df = df.dropna()
        X_processed, y_processed = preprocess_data(df)
        if X_processed.size > 0:  # Check if there's data to add
            processed_data.append((X_processed, y_processed))

    # Combine data from all files
    X_combined = np.concatenate([data[0] for data in processed_data], axis=0) if processed_data else np.array([])
    y_combined = np.concatenate([data[1] for data in processed_data], axis=0) if processed_data else np.array([])

    return X_combined, y_combined

def preprocess_data(frame):
    """
    Perform data preprocessing: feature extraction, scaling, and reshaping into trajectories.
    """
    frame['day'] = pd.to_datetime(frame['time']).dt.day
    frame['month'] = pd.to_datetime(frame['time']).dt.month
    frame['year'] = pd.to_datetime(frame['time']).dt.year
    frame["time"] = pd.to_datetime(frame["time"])
    frame["time_in_hour"] = frame["time"].dt.hour
    frame["time_in_minute"] = frame["time"].dt.minute
    frame["time_in_seconds"] = frame["time"].dt.second
    frame['plate'] = frame['plate'].astype('int64')

    X = frame[['longitude', 'latitude', 'status', 'day', 'month', 'time_in_hour', 'time_in_minute', 'time_in_seconds']].values
    Y = frame['plate']

    scaler = StandardScaler()
    
    X_reshaped = []
    y_reshaped = []

    # Process each unique plate separately
    for plate in frame['plate'].unique():
        plate_frame = frame[frame['plate'] == plate]

        # Extract the features for the current plate
        X_plate = plate_frame[['longitude', 'latitude', 'status', 'day', 'month', 'time_in_hour', 'time_in_minute', 'time_in_seconds']].values
        # Check if there is data to process
        if len(X_plate) > 0:
            X_scaled = scaler.fit_transform(X_plate)

            # Segment data into 100-step chunks, padding the last chunk if necessary
            num_chunks = len(X_scaled) // 100
            for i in range(num_chunks):
                chunk = X_scaled[i*100:(i+1)*100]
                X_reshaped.append(chunk)
                y_reshaped.append(plate)  # Appending the plate number for each chunk

            # Padding the last chunk if it's smaller than 100
            if len(X_scaled) % 100 != 0:
                last_chunk = X_scaled[num_chunks*100:]
                last_chunk_padded = np.pad(last_chunk, ((0, 100 - len(last_chunk)), (0, 0)), mode='constant')
                X_reshaped.append(last_chunk_padded)
                y_reshaped.append(plate)  # Appending the plate number for the padded chunk

    # Convert to numpy arrays
    X_reshaped = np.array(X_reshaped)
    y_reshaped = np.array(y_reshaped)
    print(X_reshaped.shape, y_reshaped.shape)
    # print(X_reshaped, y_reshaped)

    return X_reshaped, y_reshaped
