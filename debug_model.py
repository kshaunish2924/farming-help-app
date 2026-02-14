import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("plant_disease_model.keras")
model.summary()
