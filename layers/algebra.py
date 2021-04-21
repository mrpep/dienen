import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

def batch_sparse_dense_matmul(a,b):
  a_shape = tf.shape(a)
  b_shape = tf.shape(b)

  #Turn sparse matrix into supermatrix with each batch as diagonal element:
  n_instances = a_shape[0]
  a_mat_shape = a_shape[1:]
  #target_a_shape = np.array(a_mat_shape)*n_instances
  target_a_shape = tf.cast(a_mat_shape*n_instances,tf.int64)

  new_indices = tf.cast(tf.tile(tf.expand_dims(a.indices[:,0],axis=-1),(1,2)),tf.int64) * tf.cast(a_mat_shape,tf.int64) + a.indices[:,1:]
  values = a.values

  reshaped_a = tf.sparse.SparseTensor(new_indices,values,dense_shape=target_a_shape)

  #Reshape dense matrix:

  b_mat_shape = b_shape[1:]
  target_b_shape = b_mat_shape
  target_b_shape = target_b_shape*[n_instances,1]

  reshaped_b = tf.cast(tf.reshape(b,target_b_shape),tf.float32)

  result = tf.sparse.sparse_dense_matmul(reshaped_a,reshaped_b)
  return tf.reshape(result,(n_instances,a_mat_shape[0],b_mat_shape[1]))


