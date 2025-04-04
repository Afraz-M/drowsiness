import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load data
data = pd.read_csv('data/training.csv')
data.dropna(inplace=True)  # Drop rows with missing labels

# Process images
data['Image'] = data['Image'].apply(lambda im: np.fromstring(im, sep=' ', dtype=np.float32))
X = np.vstack(data['Image'].values) / 255.0  # normalize pixel values
X = X.reshape(-1, 96, 96, 1)

# Process keypoints
y = data.drop('Image', axis=1).values
y = y / 96.0  # normalize keypoints to 0-1

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# Model definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(96, 96, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.1),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(30)  # 15 keypoints * 2 (x, y)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))

# Save the model
model.save('landmark_model.h5')
print("Model saved as landmark_model.h5")
