import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers

df_test=pd.read_csv("/Users/krishrana/Python/mnist-in-csv/test.csv")
df_train=pd.read_csv("/Users/krishrana/Python/mnist-in-csv/train.csv")

labels=df_train.iloc[:, 0]
features=df_train.iloc[:, 1:785]
test=df_test.iloc[:, 0:784]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2,random_state = 1212)
print(test.shape)

X_train = X_train.as_matrix().reshape(33600, 784)
X_test = X_test.as_matrix().reshape(8400, 784)
test = test.as_matrix().reshape(28000, 784)
print((min(X_train[1]), max(X_train[1])))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
test= test.astype('float32')

X_train = X_train/255
X_test = X_test/255
test = test/255

y_train=to_categorical(y_train, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)

Inp=Input((784,))
x=(Dense(300, activation='relu'))(Inp)
x=(Dense(200, activation='relu'))(x)
x=(Dense(100, activation='relu'))(x)
x=(Dense(100, activation='relu'))(x)
output= (Dense(10, activation='softmax'))(x)

model= Model(Inp, output)
model.summary()

learning_rate=0.05
epochs=15
batch_size=100
adam=optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
model.save('digit.pb')


