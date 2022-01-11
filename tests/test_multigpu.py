#IMPORTS

import tensorflow as tf
import dienen
import pickle
import numpy as np

#Get the MNIST dataset using tf utilities:
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

#Feature scaling:
#x_train = x_train/255.
#x_test = x_test/255.

#Use the first 2000 instances for validation:
#x_val = x_train[:2000]
#y_val = y_train[:2000]

#x_train = x_train[2000:]
#y_train = y_train[2000:]

x_train = np.random.uniform(low=0,high=1,size=(10000,256,256))
x_test = np.random.uniform(low=0,high=1,size=(2000,256,256))
x_val = np.random.uniform(low=0,high=1,size=(1000,256,256))

y_train = np.random.randint(low=0,high=10,size=(10000,))
y_test = np.random.randint(low=0,high=10,size=(2000,))
y_val = np.random.randint(low=0,high=10,size=(1000,))

dnn = dienen.Model('cnn_schedules-multigpu.yaml')
dnn.set_data((x_train,y_train),validation = (x_val,y_val)) #Set the dataset
keras_model = dnn.build()
dnn.fit()


