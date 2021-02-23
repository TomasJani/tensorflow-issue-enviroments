from tensorflow.keras.layers import Embedding, Input, GRU

x = Input(shape=(None,))
x = Embedding(input_dim=50, output_dim=16, mask_zero=True)(x)
x = GRU(units=256)(x)
