import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle

# Load and preprocess dataset
anemiapath = 'Datasets/Anemia (95)'
nonanemiapath = 'Datasets/Non Anemia (123)'

anemia_image_paths = [os.path.join(anemiapath, filename) for filename in os.listdir(anemiapath)]
anemia_labels = [1] * len(anemia_image_paths)

non_anemia_image_paths = [os.path.join(nonanemiapath, filename) for filename in os.listdir(nonanemiapath)]
non_anemia_labels = [0] * len(non_anemia_image_paths)

image_paths = anemia_image_paths + non_anemia_image_paths
labels = anemia_labels + non_anemia_labels

df = pd.DataFrame({"Image": image_paths, "Label": labels})

# Split data
x_train, x_test, y_train, y_test = train_test_split(df['Image'], df['Label'], test_size=0.2, random_state=42)

# Function to extract features
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        features.append(img)
    return np.array(features)

# Prepare data for training
x_train_features = extract_features(x_train)
x_test_features = extract_features(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train_features, y_train, batch_size=32, epochs=40, validation_split=0.2)

# Save model in TensorFlow format
model.save('anemia_model.keras')
