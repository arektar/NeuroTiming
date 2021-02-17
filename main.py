import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Activation, Dropout

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import normalize

from sklearn.preprocessing import OneHotEncoder

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from tensorflow.keras.utils import to_categorical
print(to_categorical(y_train)[0, :].shape) # number of categories
print(X_train[0, :].shape) # image shape


model = Sequential()
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))

oh = OneHotEncoder(sparse=False)
oh.fit(y_train)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train/255., oh.transform(y_train), epochs=10, batch_size=32,
          validation_data=(X_test/255., oh.transform(y_test)))

model.summary()

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(X_train.shape[1:])))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train/255., to_categorical(y_train), epochs=25, batch_size=32,
          validation_data=(X_test/255., to_categorical(y_test)))
