#IMPORTS

import tensorflow as tf
import dienen
import pickle
import numpy as np

#Get the MNIST dataset using tf utilities:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

#Feature scaling:
x_train = x_train/255.
x_test = x_test/255.

#Use the first 2000 instances for validation:
x_val = x_train[:2000]
y_val = y_train[:2000]

x_train = x_train[2000:]
y_train = y_train[2000:]

results = {}
for model_name, model_config in dict(mlp='mlp.yaml', cnn='cnn_schedules.yaml').items():
    dnn = dienen.Model(model_config) #Load the model
    dnn.set_data((x_train,y_train),validation = (x_val,y_val)) #Set the dataset
    keras_model = dnn.build() #Build the model

    dnn.fit() #Train the model

    pickle.dump(dnn,open('{}.model'.format(model_name),'wb')) #Save it (dienen models are serializable)
    dnn = pickle.load(open('{}.model'.format(model_name),'rb')) #Load it
    dnn.load_weights()

    y_probs = dnn.predict(x_test) #Make some predictions
    y_pred = np.argmax(y_probs,axis=1)

    results[model_name] = 100*(y_pred == y_test).sum()/len(y_pred) #Calculate accuracy

for model_name, model_acc in results.items():
    print('{} model Test accuracy: {:.2f}%'.format(model_name,model_acc))