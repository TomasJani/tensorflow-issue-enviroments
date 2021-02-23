import tensorflow as tf
from tensorflow import keras

inference_model = tf.keras.models.load_model('./S2_models/inference_model.h5')

# Adding data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

inputs = tf.keras.Input(shape=(224, 398, 3))
x = data_augmentation(inputs)
outputs = inference_model(x, training=False)
training_model = tf.keras.Model(inputs, outputs)

print("End")

