# -*- coding: utf-8 -*-
"""Project3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B44KaeqdVK5OMLAaJlHdsS9gVSraW6C-
"""

import pandas as pd
import numpy as np
import glob
import os
import pickle
from google.colab import drive
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Mount Google Drive
drive.mount('/content/drive')

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and configured for use.")
else:
    print("No GPU available. Running on CPU.")

def process_data(traj):
    """
    Input:
        Traj: a pandas DataFrame containing one trajectory for one driver
        columns: ['plate', 'longitude', 'latitude', 'time', 'status']
    Output:
        Data: extracted features from the trajectory
    """
    traj = traj.copy()
    traj['longitude'] = traj['longitude'].astype(float)
    traj['latitude'] = traj['latitude'].astype(float)
    traj['time'] = pd.to_datetime(traj['time'])
    traj['status'] = traj['status'].astype(int)

    latitudes = traj['latitude']
    longitudes = traj['longitude']
    times = traj['time']
    statuses = traj['status']

    feature = [
        len(traj),  # trajectory length
        sum(statuses),  # total time occupied
        np.mean(latitudes),  # mean latitude
        np.mean(longitudes),  # mean longitude
        np.std(latitudes),  # std of latitude
        np.std(longitudes),  # std of longitude
        (times.max() - times.min()).total_seconds() / 3600,  # duration in hours
        np.mean(np.diff(times).astype('timedelta64[s]').astype(int)),  # average time between points in seconds
        np.mean(np.sqrt(np.diff(latitudes)**2 + np.diff(longitudes)**2))  # average speed
    ]
    return np.array(feature)

def load_and_process_data(folder_path):
    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}. Please check the folder path.")

    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        for plate in df['plate'].unique():
            traj = df[df['plate'] == plate]
            feature = process_data(traj)
            all_data.append((feature, plate))

    if not all_data:
        raise ValueError("No valid data found in the CSV files. Please check the file contents.")

    return all_data

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(data_folder):
    print(f"Looking for data in: {data_folder}")

    # load and process data
    data = load_and_process_data(data_folder)

    # split features and labels
    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)

    print(f"Processed data shape: X: {X.shape}, y: {y.shape}")

    # normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # build the model
    model = build_model((X_train.shape[1],), len(np.unique(y_encoded)))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=200, batch_size=32, callbacks=[early_stopping], verbose=1)

    # evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    return model, label_encoder, scaler

def save_model(model, label_encoder, scaler, model_path, le_path, scaler_path):
    model.save(model_path)
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    try:
        # train the model
        model, label_encoder, scaler = train_model("/content/drive/My Drive/data_5drivers")

        # save the model and related objects
        save_model(model, label_encoder, scaler,
                   '/content/drive/My Drive/taxi_driver_model.h5',
                   '/content/drive/My Drive/label_encoder.pkl',
                   '/content/drive/My Drive/scaler.pkl')

        print("model training completed and saved successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

import pickle
import numpy as np
from tensorflow.keras.models import load_model

# load the model
try:
    model = load_model('/content/drive/My Drive/taxi_driver_model.h5')
    print("Model loaded successfully.")
    print("Model summary:")
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")

# load the label encoder
try:
    with open('/content/drive/My Drive/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("\nLabel encoder loaded successfully.")
    print(f"Label encoder classes: {label_encoder.classes_}")
except Exception as e:
    print(f"Error loading label encoder: {str(e)}")

# load the scaler
try:
    with open('/content/drive/My Drive/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("\nScaler loaded successfully.")
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler scale: {scaler.scale_}")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")

# demonstrate usage
print("\nDemonstrating usage:")
sample_feature = np.random.rand(1, 9)  # Assuming 9 features
print(f"Sample feature: {sample_feature}")

scaled_feature = scaler.transform(sample_feature)
print(f"Scaled feature: {scaled_feature}")

prediction = model.predict(scaled_feature)
print(f"Raw prediction: {prediction}")

predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
print(f"Predicted driver class: {predicted_class[0]}")