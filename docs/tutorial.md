### Dienen Tutorial

#### A minimal example
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

So, in the first lines what we do is to load MNIST, and generate numpy arrays for train, validation and test sets.
Then, we create the dnn variable, using **dienen.Model(config_path)**, which constructs a Dienen model.
Then, we pass the training and validation data using **set_data()**, which receives tuples of arrays, being the first element the input and the second the target.
Calling set_data is optional, however it can be handy when we do not want to specify the input shape. If we don't call set_data, we should add a parameter to the input like:
```yaml
input_shape: [28,28]
```
telling it the dimensionality of the inputs. When set_data() is called, this parameter is automatically set by grabbing one instance of the dataset and checking its shape.

Then, we call **build()**. This method actually generates the keras model, connecting every layer and generating the tensorflow graph.
Finally, **fit()** will train the neural network with the data passed to set_data. If we don't use set_data, we can pass the data directly to fit(data=(x_train,y_train),validation_data=(x_val,y_val)).

Then, when training finishes, we can save the complete model using pickle, and then load it.
Also, we can use **predict()** to calculate outputs for given inputs. We will see later that it can also be used to extract activations from specific layers.

Finally, we calculate the accuracy and print it.

#### A bit more complex model

Let's see how to build a convolutional neural network with early stopping and saving checkpoints each epoch:

cnn_schedules.yaml
```yaml
Model:
  name: cnn
  Training:
    workers: 1
    loss: sparse_categorical_crossentropy
    metrics: [sparse_categorical_accuracy]
    optimizer:
      type: Adam
      clipvalue: 1
      learning_rate: 0.001
    schedule:
      SaveCheckpoints:
        monitor_metric: val_loss
      EarlyStopping:
        patience: 1
  Architecture:
  - Input:
      name: image
  - ExpandDims:
      axis: -1
  - Stamp:
      what:
      - Conv2D:
          kernel_size: [3,3]
          filters: [8,16,32]
          padding: SAME
      - BatchNormalization: {}
      - Activation:
          activation: relu
      - MaxPooling2D:
          pool_size: 2
      times: 3
  - Flatten: {}
  - Dense:
      name: out_probs
      units: 10
      activation: softmax
  inputs: [image]
  outputs: [out_probs]
```
In **Training** the main difference is that now we have a **schedule** key. This takes a dictionary with the name of a callback as key and its parameters as values. For example, in this case, the callbacks SaveCheckpoints and EarlyStopping will be executed at each epoch. EarlyStopping instantiates a tf.keras.callbacks.EarlyStopping class, and SaveCheckpoints is a dienen custom callback, similar to ModelCheckpoint from Tensorflow. Check its [code](../src/dienen/callbacks/save_checkpoints.py) for more details.

Then, in the architecture, a new element, called **Stamp** appears. What Stamp does is to repeat all what it is inside the key **what**, in this case 3 times.
In what, we have a Conv2D layer, followed by batch normalization, relu activation and maxpooling. This block will get repeated 3 times, however, as filters takes a list of also 3 elements, for each of the repetitions, a different filters parameters will be set. In this case, the first block will have 8 filters, while the second 16, and the third 32. This syntax allows us to build neural networks with many layers using a minimal configuration file.

Finally, the python code to train the model would be the same, but changing

```python
dnn = dienen.Model('mlp.yaml')
```

by

```python
dnn = dienen.Model('cnn_schedules.yaml')
```
Also, when doing predictions, we would like to use the weights of the best epoch instead of the last one. To do this, we can call the method **load_weights()**, which will load the weights of the best epoch. The metric used in SaveCheckpoints is used to select the epoch, and we can pass an argument **strategy** to tell if it a lower value is better or not, using 'min' or 'max'.
