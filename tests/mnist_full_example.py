import tensorflow as tf
import dienen
import pickle
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = x_train/255.
x_test = x_test/255.

x_val = x_train[:2000]
y_val = y_train[:2000]

x_train = x_train[2000:]
y_train = y_train[2000:]

results = {}
for model_name, model_config in dict(mlp='mlp.yaml', cnn='cnn.yaml').items():
    dnn = dienen.Model(model_config)
    dnn.set_data((x_train,y_train),validation = (x_val,y_val))
    keras_model = dnn.build()

    dnn.fit()

    pickle.dump(dnn,open('{}.model'.format(model_name),'wb'))
    dnn = pickle.load(open('{}.model'.format(model_name),'rb'))

    y_probs = dnn.predict(x_test)
    y_pred = np.argmax(y_probs,axis=1)

    results[model_name] = 100*(y_pred == y_test).sum()/len(y_pred)

for model_name, model_acc in results.items():
    print('{} model Test accuracy: {:.2f}%'.format(model_name,model_acc))