import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

rng = np.random.RandomState(0)
batch_size = 32
n_samples, n_features = batch_size * 10, 5
X = rng.normal(size=(n_samples, n_features))
y = rng.randint(low=0, high=2, size=X.shape[0])

model = Sequential([Dense(1, input_shape=(n_features,),
                          activation="sigmoid")])
model.compile(optimizer="adam", loss='binary_crossentropy',
              metrics=['accuracy'])

print("model.test_on_batch without model.evaluate")
loss_1 = 0
for i in range(3):
    loss, acc = model.test_on_batch(X[:batch_size], y[:batch_size])
    if i == 0:
        loss_1 = loss
    print(loss, acc)

# Correct
normal_loss, normal_acc = model.evaluate(
    X, y, batch_size=batch_size, verbose=0)

loss_2 = 0
print("model.test_on_batch *after* model.evaluate")
for i in range(3):
    loss, acc = model.test_on_batch(X[:batch_size], y[:batch_size])
    if i == 0:
        loss_2 = loss
    print(loss_2, acc)


assert loss_1 == loss_2
