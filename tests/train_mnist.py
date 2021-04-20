import tensorflow as tf
import dienen

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = x_train/255.
x_test = x_test/255.

dnn = dienen.Model('mlp.yaml')





