import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------------------
# SETTINGS
# -------------------------------
DATASET_DIR = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
MODEL_NAME = "plant_disease_model.h5"

# -------------------------------
# LOAD IMAGES USING A GENERATOR
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,          # 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

# -------------------------------
# SAVE class indices → classes.json
# -------------------------------
class_names = list(train_gen.class_indices.keys())

with open("classes.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("Classes saved to classes.json")
print(class_names)

# -------------------------------
# BUILD TRANSFER-LEARNING MODEL
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze layers

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# TRAIN THE MODEL
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------------------------------
# SAVE THE MODEL
# -------------------------------
model.save(MODEL_NAME)
print(f"\nModel saved as {MODEL_NAME}")
