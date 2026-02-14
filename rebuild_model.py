import tensorflow as tf
import keras

MODEL_H5 = "plant_disease_model.h5"
OUTPUT = "plant_disease_model_clean.keras"
IMG_SIZE = 224

# ---- 1. Rebuild clean MobileNetV2 architecture ----
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=None
)

x = keras.layers.GlobalAveragePooling2D()(base.output)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(13, activation="softmax")(x)   # you have 13 classes

clean_model = keras.Model(inputs=base.input, outputs=outputs)
clean_model.build((None, IMG_SIZE, IMG_SIZE, 3))

print("Empty clean model built.")

# ---- 2. Load weights from broken H5 model ----
print("Loading weights from H5 file...")
temp_model = keras.models.load_model(MODEL_H5, compile=False)

clean_model.set_weights(temp_model.get_weights())

# ---- 3. Save the new clean Keras v3 model ----
print("Saving clean Keras model...")
clean_model.save(OUTPUT)

print("\nDONE! Saved clean model as:", OUTPUT)
