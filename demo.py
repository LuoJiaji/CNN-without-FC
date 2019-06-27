from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Convolution2D, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

input_data = Input(shape=(28,28,1))
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(input_data)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Conv2D(128, (7, 7), activation='relu', strides=(2, 2), name='temp')(x)

x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(10, activation='softmax', name='fc2')(x)

model = Model(input_data, x)
model.summary()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)/255.
X_test = X_test.reshape(-1, 28, 28, 1)/255.
y_train = np_utils.to_categorical(y_train, num_classes= 10)
y_test = np_utils.to_categorical(y_test, num_classes= 10)

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64)


loss, accuracy = model.evaluate(X_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)