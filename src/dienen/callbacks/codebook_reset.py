from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

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

class CodebookWarmupV2(Callback):
    def __init__(self, encoder_pretrain_steps=50000, codebook_only_train_steps=10000, codebook_init = 'pca', codebook_layer=None, init_dataset=None, estimation_points=10000):
        self.encoder_pretrain_steps = encoder_pretrain_steps
        self.codebook_only_train_steps = codebook_only_train_steps
        self.codebook_init = codebook_init
        self.codebook_layer = codebook_layer
        self.init_dataset = init_dataset
        self.estimation_points = estimation_points
        self.step = 0

    def pca_init(self):
        from sklearn.decomposition import PCA
        import kmeans1d

        #Get the codebook inputs for the init_dataset:
        codebook_layer_in = self.codebook_layer.input
        n_batches = self.estimation_points//self.data.batch_size
        pred_fn = tf.keras.backend.function(inputs=self.model.inputs,outputs=[codebook_layer_in])
        encoded_data = [pred_fn(self.init_dataset.__getitem__(i)[0])[0] for i in range(n_batches)]
        encoded_data = np.concatenate(encoded_data)
        encoded_data = np.reshape(encoded_data,(-1,encoded_data.shape[-1]))

        #Apply PCA:
        pca_model = PCA(n_components = self.codebook_layer.groups)
        encoded_low = pca_model.fit_transform(encoded_data)

        #Kmeans over each principal component
        groups_centroids = []
        for component_values in encoded_low.T:
            clusters,centroids = kmeans1d.cluster(component_values, self.codebook_layer.codes_per_group)
            groups_centroids.append(centroids)
        groups_centroids = np.array(groups_centroids)[:,:,np.newaxis]
        codebook = groups_centroids*np.transpose(pca_model.components_[:,:,np.newaxis],(0,2,1))

        return codebook

    def on_train_begin(self,logs):
        self.codebook_layer = self.model.get_layer(self.codebook_layer)
        if self.encoder_pretrain_steps > 0:
            #Skip quantization layer:
            cb_w = self.codebook_layer.get_weights()
            cb_w[-2] = np.float32(0.0)
            cb_w[-1] = np.float32(1.0)
            self.codebook_layer.set_weights(cb_w)
            from IPython import embed
            embed()
        
    def on_batch_end(self,batch,logs):
        if self.step == self.encoder_pretrain_steps:
            #Initialize codebook and activate it:
            cb_w = self.codebook_layer.get_weights()
            if self.codebook_init == 'pca':
                codebook_weights = self.pca_init()
            cb_w[0] = codebook_weights
            cb_w[-2] = np.float32(1.0)
            cb_w[-1] = np.float32(0.0)
            self.codebook_layer.set_weights(cb_w)
            #Freeze other layers:
            self.original_trainable_states = [l.trainable for l in self.model.layers]
            for l in self.model.layers:
                if l.name != self.codebook_layer.name:
                    l.trainable = False
        elif self.step == self.encoder_pretrain_steps + self.codebook_only_train_steps:
            #Restore trainable states
            for l, t in zip(self.model.layers, self.original_trainable_states):
                l.trainable = t

        self.step += 1

    def on_epoch_end(self,epoch,logs):
        pass

class CodebookWarmup(Callback):
    def __init__(self, pretrain_steps=15000, warmup_steps=5000, codebook_layer=None,estimation_points=2000):
        self.pretrain_steps = pretrain_steps
        self.warmup_steps = warmup_steps
        self.step = 0
        self.codebook_layer = codebook_layer
        self.estimation_points = estimation_points
    
    def on_train_begin(self,logs):
        self.codebook_layer = self.model.get_layer(self.codebook_layer)
        if self.pretrain_steps>0:
            cb_w = self.codebook_layer.get_weights()
            cb_w[-2] = np.float32(0.0)
            cb_w[-1] = np.float32(1.0)
            self.codebook_layer.set_weights(cb_w)

    def on_batch_end(self,batch,logs):
        if self.step == self.pretrain_steps:
            #Gather codebook inputs:
            codebook_layer_in = self.codebook_layer.input
            n_batches = self.estimation_points//self.data.batch_size
            pred_fn = tf.keras.backend.function(inputs=self.model.inputs,outputs=[codebook_layer_in])
            encoded_data = [pred_fn(self.data.__getitem__(i)[0])[0] for i in range(n_batches)]
            encoded_data = np.concatenate(encoded_data)
            encoded_data = np.reshape(encoded_data,(-1,encoded_data.shape[-1]))
            #Reduce dim
            pca_dim = self.codebook_layer.vq_dim
            n_clusters = self.codebook_layer.codes_per_group
            n_codebooks = self.codebook_layer.groups
            pca_model = PCA(n_components=pca_dim)
            encoded_data = pca_model.fit_transform(encoded_data)
            #Clustering (ver como agrupar para hacer clusters distintos por grupo)
            kmeans = MiniBatchKMeans(n_clusters=n_clusters)
            kmeans.fit(encoded_data)
            #Asignar centroides a los codebooks
            cb_w = self.codebook_layer.get_weights()
            cb_w[0] = np.tile(kmeans.cluster_centers_,(8,1,1))
            self.codebook_layer.set_weights(cb_w)
        if self.step > self.pretrain_steps and self.step < self.pretrain_steps + self.warmup_steps:
            codebook_weight = (self.step - self.pretrain_steps)/self.warmup_steps
            cb_w = self.codebook_layer.get_weights()
            cb_w[-2] = np.float32(codebook_weight)
            cb_w[-1] = np.float32(1.0 - codebook_weight)
            self.codebook_layer.set_weights(cb_w)
        elif self.step == self.pretrain_steps + self.warmup_steps:
            cb_w = self.codebook_layer.get_weights()
            cb_w[-2] = np.float32(1.0)
            cb_w[-1] = np.float32(0)
            self.codebook_layer.set_weights(cb_w)

        self.step += 1

    def on_epoch_end(self,epoch,logs):
        pass