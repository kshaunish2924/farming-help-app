import keras
import tensorflow as tf
from keras.layers import Layer, Rescaling
from keras.saving import custom_object_scope

MODEL_PATH = "plant_disease_model.h5"
OUTPUT_PATH = "plant_disease_model.keras"

# Fake TrueDivide layer for loading legacy MobileNetV2 models
class TrueDivide(Layer):
    def __init__(self, **kwargs):
        super(TrueDivide, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs  # identity operation

custom_objects = {
    "TrueDivide": TrueDivide,
    "Rescaling": Rescaling
}

print("Loading H5 model using custom_object_scope...")

with custom_object_scope(custom_objects):
    model = keras.models.load_model(MODEL_PATH, compile=False)

print("Saving in new Keras v3 format...")
model.save(OUTPUT_PATH)

print("Done! Saved as:", OUTPUT_PATH)
