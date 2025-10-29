import cv2
from pathlib import Path
import os
import numpy as np 
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

# Dataset path
TRAIN_DIR = Path.cwd() / 'Training'
TEST_DIR = Path.cwd() / 'Testing'

# Class names (exactly as folder names)
class_names = ['glioma_tumor', 'meningioma_tumor','no_tumor','pituitary_tumor']

# Image size
INPUT_SIZE = (64, 64)

# Helper to load images
def load_images_from_folder(folder_path, label):
    dataset = []
    labels = []
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(str(folder_path / image_name))
            image = Image.fromarray(image, 'RGB').resize(INPUT_SIZE)
            dataset.append(np.array(image))
            labels.append(label)
    return dataset, labels

# Load training and testing data
dataset = []
labels = []

for idx, class_name in enumerate(class_names):
    class_folder = TRAIN_DIR / class_name
    imgs, lbls = load_images_from_folder(class_folder, idx)
    dataset.extend(imgs)
    labels.extend(lbls)

for idx, class_name in enumerate(class_names):
    class_folder = TEST_DIR / class_name
    imgs, lbls = load_images_from_folder(class_folder, idx)
    dataset.extend(imgs)
    labels.extend(lbls)

# Convert to numpy arrays
dataset = np.array(dataset)
labels = np.array(labels)

# Normalize and one-hot encode
dataset = normalize(dataset, axis=1)
labels = to_categorical(labels, num_classes=len(class_names))

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

# Build CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test), shuffle=True)

# Save model
model.save('BrainTumorModel_MultiClass.h5')
print("✅ Model saved as BrainTumorModel_MultiClass.h5")
