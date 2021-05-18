from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans

import numpy as np

class CodebookReset(Callback):
    def __init__(self,technique='kmeans',layer=None,freq=1000,unit='step',estimation_points=5000):
        self.technique = technique
        self.layer = layer
        self.freq = freq
        self.unit = unit
        self.step = 0
        self.epoch = 0
        self.estimation_points = estimation_points

    def kmeans_init(self):
        codebook_layer = self.model.get_layer(self.layer)
        codebook_inputs = codebook_layer.input

        pred_fn = tf.keras.backend.function(inputs=self.model.inputs,outputs=[codebook_inputs])
        n_batches = self.estimation_points // self.data.batch_size + 1

        data = np.concatenate([pred_fn(self.data.__getitem__(i)[0])[0] for i in range(n_batches)],axis=0)[:self.estimation_points]
        data = np.reshape(data,[-1,data.shape[-1]])
        n_codes = codebook_layer.get_weights()[0].shape[0]
        kmeans = MiniBatchKMeans(n_clusters=n_codes)

        kmeans.fit(data)
        codebook_layer.set_weights([kmeans.cluster_centers_])

    def on_train_begin(self, logs):
        self.kmeans_init()

    def on_batch_end(self,batch,logs):
        self.step += 1
        if self.step % self.freq == 0 and self.unit == 'step':
            self.kmeans_init()

    def on_epoch_end(self,epoch,logs):
        self.epoch += 1
        if self.epoch % self.freq == 0 and self.unit == 'epoch':
            self.kmeans_init()