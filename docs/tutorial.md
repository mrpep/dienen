### Dienen Tutorial

Let's do the classical ML example of MNIST digits recognition. First we are going to build a very simple model consisting of a Multi Layer Perceptron (MLP). The configuration file for the model is this .yaml document:

mlp.yaml

```yaml
Model:
  name: mlp
  Training:
    workers: 1
    loss: sparse_categorical_crossentropy
    metrics: [sparse_categorical_accuracy]
    optimizer:
      type: Adam
      clipvalue: 1
      learning_rate: 0.001
  Architecture:
  - Input:
      name: image
  - Flatten: {}
  - Dense:
      name: hidden_layer
      units: 128
      activation: relu
  - Dense:
      name: out_probs
      units: 10
      activation: softmax
  inputs: [image]
  outputs: [out_probs]
```

Let's analyze what all of this means.
First of all there is a key called **Model**, which is the parent of everything in the config.
We can give the model a name (in this case mlp), and then there is the **Training** key.

In the **Training** key we can specify all the things related to how the model will be trained.
The **loss** parameter can be a plain string, like in this case, or a dictionary, in the case that the
loss function we refer to is a class inherited from tf.keras.losses.Loss, and the dictionary should contain the key type, where the class name is
indicated, and then any parameters we want to pass to the Loss initializer. The same logic applies for the **metrics** and **optimizer**. In the case of the metrics, we can pass a list of plain strings or dictionaries, if we want to monitor multiple metrics. In the case of the optimizer, we are instantiating tf.keras.optimizers.Adam, and passing the clipvalue and learning_rate as arguments in the constructor.

The **Architecture** key, is where we define the layers of the neural network. The layers are given in a list. At least one **Input** layer should be specified.
In this case, we give it the name "image". Then, we have the Flatten layer, which will be a tf.keras.layers.Flatten layer. In this case, it doesn't receive any parameter so we pass an empty dictionary. The layer will be given a default name, and its **input** parameter set to the last layer (in this case, image).
Then, we have two Dense layers, which are tf.keras.layers.Dense layers. In general, you will be able to access any layer in tf.keras.layers by using its name as key. Also, you will be able to access to some Dienen custom layers, which are [here](../src/dienen/layers).
The output is called out_probs and has 10 units (neurons), which correspond to the 10 digits we want to classify. Finally, we need to tell dienen which layers are inputs and which outputs. We pass their names in the **inputs** and **outputs** keys.

Now, we are halfway there. It is time to write the code that loads this configuration file, prepares the data and trains the model:

train_mnist.py
```python
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

dnn = dienen.Model('mlp.yaml') #Load the model
dnn.set_data((x_train,y_train),validation = (x_val,y_val)) #Set the dataset
keras_model = dnn.build() #Build the model

dnn.fit() #Train the model

pickle.dump(dnn,open('mlp.model','wb')) #Save it (dienen models are serializable)
dnn = pickle.load(open('mlp.model','rb')) #Load it

y_probs = dnn.predict(x_test) #Make some predictions
y_pred = np.argmax(y_probs,axis=1)

acc = 100*(y_pred == y_test).sum()/len(y_pred) #Calculate accuracy
print('MLP model Test accuracy: {:.2f}%'.format(acc))
```


