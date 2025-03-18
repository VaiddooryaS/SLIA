import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Dataset path
DATASET_PATH = "C:/Users/Administrator/Documents/vidhu/uniii/python/slia/dataset/Root/"
IMAGE_SIZE = (64, 64)
LIMIT_PER_CLASS = 200  # 200 images per letter from each type

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def process_image(image_path):
    """Loads and processes an image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Warning: Could not read image {image_path}")
        return None
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.array(image) / 255.0  # Normalize


def load_data():
    """Loads images from both gesture types, ensuring 200 images per letter from each."""
    data, labels = [], []
    gesture_types = ["Type_01_(Raw_Gesture)", "Type_02_(Keypoint Based)"]

    for gesture_type in gesture_types:
        gesture_path = os.path.join(DATASET_PATH, gesture_type)

        for category in os.listdir(gesture_path):
            category_path = os.path.join(gesture_path, category)
            if not os.path.isdir(category_path):
                continue

            images = os.listdir(category_path)[:LIMIT_PER_CLASS]  # Take only first 200 images
            for img_name in images:
                img_path = os.path.join(category_path, img_name)
                processed_img = process_image(img_path)
                if processed_img is not None:
                    data.append(processed_img)
                    labels.append(category)  # Category is the letter (A-Z, SPACE, DEL)

    return np.array(data), np.array(labels)


# Load dataset
X, y = load_data()
X = X.reshape(-1, 64, 64, 1)  # Reshape for CNN input

# Dynamically get class labels
unique_labels = sorted(set(y))
num_classes = len(unique_labels)

# Create mapping for labels
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y = tf.keras.utils.to_categorical([label_to_index[label] for label in y], num_classes=num_classes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Uses correct number of classes
])

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save Model
model.save("models/sign_language_model.h5")

# Print summary
print(f"✅ Model trained with {num_classes} classes: {unique_labels}")
