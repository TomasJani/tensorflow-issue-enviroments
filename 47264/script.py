import tensorflow as tf

module = tf.Module()
module.submodule = tf.Module()
module.submodule.var = tf.Variable(1.0)
assert module.trainable_variables == (module.submodule.var,)  # as expected

layer = tf.keras.layers.Layer()
assert isinstance(layer, tf.Module)  # passes
layer.sublayer = tf.keras.layers.Layer()
layer.sublayer.var = tf.Variable(1.0)
assert layer.trainable_variables == [layer.sublayer.var]  # acceptable

layer = tf.keras.layers.Layer()
layer.submodule = tf.Module()
layer.submodule.var = tf.Variable(1.0)
assert list(layer.trainable_variables) == [layer.submodule.var]  # FAILS
