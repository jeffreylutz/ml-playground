from keras.models import Sequential
from keras.callbacks import History
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

Xtest = np.array([[0,0],[0,1],[1,0],[1,1]])
ytest = np.array([[0],[1],[1],[0]])
X = np.array([[0,0]],"float32")
y = np.array([[0]],"float32")
for i in range(1,40):
    if i%4 == 0:
        X = np.append(X,[[0,0]] ,axis=0)
        y = np.append(y,[[0]] ,axis=0)
    if i%4 == 1:
        X = np.append(X,[[0,1]] ,axis=0)
        y = np.append(y,[[1]] ,axis=0)
    if i%4 == 2:
        X = np.append(X,[[1,0]] ,axis=0)
        y = np.append(y,[[1]] ,axis=0)
    if i%4 == 3:
        X = np.append(X,[[1,1]] ,axis=0)
        y = np.append(y,[[0]] ,axis=0)

print(X.shape)
adam = Adam(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=adam)
# model.compile(loss='binary_crossentropy', optimizer='adam')
# model.compile(loss='mse', optimizer='adam')
sgd = SGD(lr=0.1)
# model.compile(loss='binary_crossentropy', optimizer=sgd)
# history = model.fit(X, y, epochs=200, batch_size=1, verbose=0)
history = model.fit(X, y, validation_split=.2, epochs=20, batch_size=1, verbose=0)

loss = history.history['loss']
val_loss = history.history['val_loss']

print(Xtest,model.predict_classes(Xtest))

plt.plot(loss)
plt.plot(val_loss)
plt.show()
