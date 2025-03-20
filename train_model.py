import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import shutil

# ✅ Paths
DATASET_PATH = r"C:\Users\Administrator\Documents\vidhu\uniii\python\slia\dataset\Root\Type_02_(Keypoint Based)"
MODEL_SAVE_PATH = r"C:\Users\Administrator\Documents\vidhu\uniii\python\slia\models\new_model.h5"
TEMP_TRAIN_PATH = r"C:\Users\Administrator\Documents\vidhu\uniii\python\slia\dataset\temp_train"

# ✅ Fixed class labels
CLASS_NAMES = ['A', 'B', 'C', 'D', 'DEL', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
               'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# ✅ Ensure only the valid classes are used
def filter_and_balance_dataset():
    if os.path.exists(TEMP_TRAIN_PATH):
        shutil.rmtree(TEMP_TRAIN_PATH)
    os.makedirs(TEMP_TRAIN_PATH, exist_ok=True)

    for label in CLASS_NAMES:
        original_class_path = os.path.join(DATASET_PATH, label)
        temp_class_path = os.path.join(TEMP_TRAIN_PATH, label)

        if not os.path.exists(original_class_path):
            print(f"Skipping {label}, folder not found in dataset.")
            continue

        os.makedirs(temp_class_path, exist_ok=True)
        images = os.listdir(original_class_path)
        random.shuffle(images)
        selected_images = images[:800]  # Pick 800 random images

        for img in selected_images:
            src = os.path.join(original_class_path, img)
            dst = os.path.join(temp_class_path, img)
            shutil.copy(src, dst)

    print("✅ Dataset balanced with 800 images per class.")


# ✅ Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


# ✅ Load and Prepare Data
def get_data_generators():
    train_gen = datagen.flow_from_directory(
        TEMP_TRAIN_PATH,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        TEMP_TRAIN_PATH,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen


# ✅ Define CNN Model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASS_NAMES), activation='softmax')  # 28 output classes
    ])

    model.compile(
        optimizer=AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ✅ Training Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)


# ✅ Train Model
def train():
    filter_and_balance_dataset()
    train_gen, val_gen = get_data_generators()
    model = create_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop, reduce_lr]
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
