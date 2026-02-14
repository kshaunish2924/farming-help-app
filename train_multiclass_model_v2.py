import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json
import os

# -------------------------------
# SETTINGS
# -------------------------------
DATASET_DIR = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12  # you can increase later
MODEL_NAME = "plant_disease_model.keras"  # NOTE: .keras format

# -------------------------------
# DATA LOADERS
# -------------------------------
# We use preprocess_input in the DATA PIPELINE, not inside the model.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
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
# SAVE CLASSES
# -------------------------------
class_names = list(train_gen.class_indices.keys())
with open("classes.json", "w") as f:
    json.dump(class_names, f, indent=4)
print("Classes:", class_names)

# -------------------------------
# BUILD MODEL (NO PREPROCESSING OPS INSIDE)
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze backbone

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
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
# TRAIN
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------------------------------
# SAVE CLEAN .KERAS MODEL
# -------------------------------
model.save(MODEL_NAME)
print(f"\nSaved model as: {MODEL_NAME}")
