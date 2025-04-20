import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess images
def load_data(data_dir):
    images = []
    labels = []
    for label in ["open", "closed"]:
        path = os.path.join(train_dir, label)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (24, 24))
            images.append(img)
            labels.append(0 if label == "closed" else 1)  # Assign labels: 0 for closed, 1 for open
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels

# Paths to train data directory (this is where the "train" folder exists)
train_dir = "C:/Projects/DDS/Scripts/dataset/train"

# Load train dataset (Only train data)
X_train, y_train = load_data(train_dir)

# Reshape data and one-hot encode labels
X_train = X_train.reshape(X_train.shape[0], 24, 24, 1)
y_train = to_categorical(y_train, 2)

# Data augmentation for training set
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using only the training data
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=15, verbose=1)

# Save the trained model
model.save("model.h5")
print("Model saved as model.h5")

