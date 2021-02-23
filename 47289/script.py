import numpy as np
import tensorflow as tf
layer = tf.keras.layers.ZeroPadding2D(padding=420958214)
layer(np.ones((0, 4, 4, 4)))
