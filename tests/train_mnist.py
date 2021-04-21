import tensorflow as tf
import dienen

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = x_train/255.
x_test = x_test/255.

x_val = x_train[:2000]
y_val = y_train[:2000]

x_train = x_train[2000:]
y_train = y_train[2000:]

dnn = dienen.Model('mlp.yaml')
dnn.set_data((x_train,y_train),validation = (x_val,y_val))
keras_model = dnn.build()
keras_model.summary()

dnn.fit()







