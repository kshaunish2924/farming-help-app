import tensorflow as tf
from tensorflow import keras

m = keras.models.load_model("plant_disease_model.keras")

for i, layer in enumerate(m.layers):
    print(i, layer.name, layer.__class__.__name__)
    if isinstance(layer, keras.Model):
        print("  SUBMODEL:")
        for j, sub in enumerate(layer.layers):
            print("    ", j, sub.name, sub.__class__.__name__)
