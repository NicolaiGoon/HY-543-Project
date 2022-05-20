import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


x_train = tf.keras.applications.densenet.preprocess_input( x_train, data_format=None)
x_val = tf.keras.applications.densenet.preprocess_input( x_val, data_format=None)


model = tf.keras.applications.densenet.DenseNet121(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation='softmax'
)


model.compile(loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), batch_size=32, verbose=1)
