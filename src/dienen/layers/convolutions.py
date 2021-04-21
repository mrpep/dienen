import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
from .algebra import *

class DynamicConvolution(tfkl.Layer):
    """
    Wu, F., Fan, A., Baevski, A., Dauphin, Y. N., & Auli, M. (2019). Pay less attention with lightweight and dynamic convolutions. arXiv preprint arXiv:1901.10430.
    """

    def __init__(self, kernel_size, H, softmax_on_kernel=True, dropconnect=0.2, kernel_estimator='conv', kernel_estimator_kernel_size=1,kernel_estimator_activation='linear',name=None):
        super(DynamicConvolution, self).__init__(name=name, trainable=True)
        self.kernel_size = kernel_size
        self.H = H
        self.softmax_on_kernel = softmax_on_kernel
        self.kernel_estimator = kernel_estimator
        self.kernel_estimator_kernel_size = kernel_estimator_kernel_size
        self.kernel_estimator_activation = kernel_estimator_activation
        self.dropconnect = dropconnect

    def get_band_matrix_indexs_and_mask(self, seqlen, kernel_size):
        # Calculate from kernel size the offsets from diagonal the band will have.
        # In this case it is done so that the kernel is centered.
        if kernel_size % 2 == 0:
            offset = [-(kernel_size//2), kernel_size//2-1]
        else:
            offset = [-(kernel_size//2), kernel_size//2]

        # Mask values that fall outside the matrix
        mask_indexs = np.ones((seqlen, offset[1]-offset[0]+1))
        for i in range(mask_indexs.shape[0]):
            for j in range(mask_indexs.shape[1]):
                if i + j + offset[0] < 0 or i + j + offset[0] >= seqlen:
                    mask_indexs[i, j] = 0
        mask_indexs = mask_indexs.flatten().astype(bool)
        #mask_indexs = np.tile(mask_indexs,reps=(batch_size))

        # Indexs of the diagonal band
        #sparse_diagonal_indexs = np.concatenate([np.array((j*np.ones((min(i + offset[1] + 1,seqlen) - max(0,i + offset[0]),)),i*np.ones((min(i + offset[1] + 1,seqlen) - max(0,i + offset[0]),)),np.arange(max(0,i + offset[0]),min(i + offset[1] + 1,seqlen)))).T for j in range(batch_size) for i in range(seqlen)])
        sparse_diagonal_indexs = np.concatenate([np.array((i*np.ones((min(i + offset[1] + 1, seqlen) - max(
            0, i + offset[0]),)), np.arange(max(0, i + offset[0]), min(i + offset[1] + 1, seqlen)))).T for i in range(seqlen)])

        return mask_indexs, sparse_diagonal_indexs

    def build(self, input_shape):
        #self.batch_size = input_shape[0]
        self.seqlen = input_shape[1]
        # para evitar lo del batchsize podria asumir batch size 1 y en call donde el bs esta disponible, hacer un sparse concat
        self.mask_indexs, self.sparse_diagonal_indexs = self.get_band_matrix_indexs_and_mask(
            self.seqlen, self.kernel_size)
        #self.conv_layer = tfkl.Conv1D(self.kernel_size*self.H,1)
        if self.kernel_estimator == 'conv':
            self.estimator_layer = tfkl.Conv1D(self.kernel_size*self.H,kernel_size=self.kernel_estimator_kernel_size,activation=self.kernel_estimator_activation,padding='SAME')
        elif self.kernel_estimator == 'gru':
            self.estimator_layer = tfkl.GRU(
                self.kernel_size*self.H, return_sequences=True, activation=self.kernel_estimator_activation)
        elif self.kernel_estimator == 'lstm':
            self.estimator_layer = tfkl.LSTM(
                self.kernel_size*self.H, return_sequences=True, activation=self.kernel_estimator_activation)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        n_ch = input_shape[2]

        # This returns position-wise kernels
        predicted_kernels = self.estimator_layer(x)
        predicted_kernels = tf.reshape(
            predicted_kernels, (batch_size, self.seqlen, self.kernel_size, self.H))
        if self.softmax_on_kernel:
            predicted_kernels = tf.nn.softmax(predicted_kernels, axis=-2)
        if self.dropconnect>0:
            predicted_kernels = tfkl.Dropout(self.dropconnect)(predicted_kernels)
        predicted_kernels = tf.transpose(predicted_kernels, (0, 3, 1, 2))
        predicted_kernels = tf.reshape(
            predicted_kernels, (batch_size*self.H, self.seqlen, self.kernel_size))

        # batch_size = tf.shape(predicted_kernels[0] #Use batch size to rearrange sparse band matrix and boolean mask
        batch_dim = tf.expand_dims(tf.repeat(tf.range(
            0, batch_size*self.H, dtype=tf.int64), self.sparse_diagonal_indexs.shape[0]), axis=-1)
        idxs_repeated = tf.tile(tf.cast(self.sparse_diagonal_indexs, tf.int64), [
                                batch_size*self.H, 1])

        new_sparse_idxs = tf.concat([batch_dim, idxs_repeated], axis=-1)
        mask_indexs = tf.tile(self.mask_indexs, [batch_size*self.H])

        sparse_band_matrix = tf.sparse.SparseTensor(indices=new_sparse_idxs, values=tf.boolean_mask(tf.reshape(
            predicted_kernels, [-1]), mask_indexs), dense_shape=[batch_size*self.H, self.seqlen, self.seqlen])

        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (batch_size*self.H, n_ch//self.H, self.seqlen))
        x = tf.transpose(x, (0, 2, 1))

        result = batch_sparse_dense_matmul(sparse_band_matrix, x)
        result = tf.reshape(
            result, (batch_size, self.H, self.seqlen, n_ch//self.H))
        result = tf.transpose(result, (0, 2, 3, 1))

        return tf.reshape(result, self.compute_output_shape(input_shape))

class GLU(tfkl.Layer):
  def __init__(self,n_filters,kernel_size,kernel_size_gate = None,activation='tanh',gate_activation='sigmoid',padding='same',name=None):
    super(GLU,self).__init__(name=name)
    self.conv = tfkl.Conv1D(n_filters,kernel_size,activation=activation,padding=padding)
    if kernel_size_gate is None:
      kernel_size_gate = kernel_size
    self.gate_conv = tfkl.Conv1D(n_filters,kernel_size_gate,activation=gate_activation,padding=padding)
    self.multiply = tfkl.Multiply()
  
  def call(self,x):
    y = self.conv(x)
    y_mask = self.gate_conv(x)
    return self.multiply([y,y_mask])

class SqueezeAndExcite2D(tfkl.Layer):
  def __init__(self,ratio,name=None, trainable=True):
    super(SqueezeAndExcite2D,self).__init__(name=name, trainable=trainable)
    self.ratio = ratio

  def build(self, input_shape):
    filters = input_shape[-1]
    self.ga = tfkl.GlobalAveragePooling2D()
    self.rs = tfkl.Reshape([1,1,filters])
    self.d1 = tfkl.Dense(filters // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)
    self.d2 = tfkl.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
  
  def call(self,x):
    se = self.ga(x) #Squeeze
    se = self.rs(se)
    se = self.d1(se) #Excitation
    se = self.d2(se)
    se = tfkl.Multiply()([se,x]) #Scaling
    return se 

  def get_config(self):
    config = super().get_config().copy()
    config.update({
            'ratio': self.ratio
        })
    return config

  def compute_output_shape(self,input_shape):
      return input_shape