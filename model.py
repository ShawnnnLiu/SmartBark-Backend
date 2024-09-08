import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import layers, models

data_path = '/path/to/UrbanSound8K/audio/'

class_mapping = {
    'air_conditioner': 0,
    'car_horn': 1,
    'children_playing': 2,
    'dog_bark': 3,
    'drilling': 4,
    'engine_idling': 5,
    'gun_shot': 6,
    'jackhammer': 7,
    'siren': 8,
    'street_music': 9
}

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, resample=True, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40, fmax=8000)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

features = []
labels = []

for folder_name, class_id in class_mapping.items():
    folder_path = os.path.join(data_path, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        mel_spectrogram_db = extract_features(file_path)
        
        if mel_spectrogram_db is not None:
            features.append(mel_spectrogram_db)
            labels.append(class_id)

X = np.array(features)
y = np.array(labels)

X = X / np.max(X)

y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("Data preprocessing complete.")


model = models.Sequential()

# Input shape should match the shape of the extracted Mel-spectrograms
input_shape = (X_train.shape[1], X_train.shape[2], 1)

# Add layers to the model
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 output classes


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

model.save("my_model.h5")
