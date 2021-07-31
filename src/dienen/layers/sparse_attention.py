import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from .algebra import batch_sparse_dense_matmul

def row_attention_pattern(block_size=4, N=32):
    lookup_indices = np.zeros((N, block_size),dtype=int)
    lookup_mask = np.zeros((N,block_size))
    sparse_indices = []
    for i in range(N):
        for j in range(i%block_size+1):
            lookup_indices[i,j]=abs(j-i)
            lookup_mask[i,j]=1
            sparse_indices.append([i,i - i%block_size + j])
    return lookup_indices, lookup_mask, np.array(sparse_indices)

def col_attention_pattern(block_size=4, N=32):
    lookup_indices = np.zeros((N, N//block_size),dtype=int)
    lookup_mask = np.zeros((N,N//block_size))
    sparse_indices = []
    for i in range(N):
        for j in range(i//block_size + 1):
            lookup_indices[i,j]=i-j*block_size
            lookup_mask[i,j]=1
            sparse_indices.append([i,j*block_size + i%block_size])
    return lookup_indices, lookup_mask, np.array(sparse_indices)

def previous_row_attention_pattern(block_size=4, N=32):
    lookup_indices = np.zeros((N, block_size+1),dtype=int)
    lookup_mask = np.zeros((N,block_size+1))
    sparse_indices = []
    for i in range(N):
        for j in range(block_size*min(i//block_size,1)):
            lookup_indices[i,j]=i-i%block_size-block_size+j
            lookup_mask[i,j]=1
            sparse_indices.append([i,i-i%block_size-block_size+j])
        lookup_indices[i,block_size]=i
        lookup_mask[i,block_size]=1
        sparse_indices.append([i,i])
    
    return lookup_indices, lookup_mask, np.array(sparse_indices)

class BlockSparseProductAttention(tfkl.Layer):
    def __init__(self,block_size=16,sparse_pattern='row',apply_softmax=True,name=None,trainable=False):
        super(BlockSparseProductAttention,self).__init__(name=name)
        self.block_size = block_size
        self.apply_softmax = apply_softmax
        self.sparse_pattern = sparse_pattern
        self.T = None

    def build(self,x_shape):
        if self.T is None:
            self.T = tf.cast(x_shape[0][2],dtype=tf.int32)

        if self.sparse_pattern == 'row':
            lookup_indices, lookup_mask, sparse_indices = row_attention_pattern(block_size=self.block_size, N = self.T)
        elif self.sparse_pattern == 'col':
            lookup_indices, lookup_mask, sparse_indices = col_attention_pattern(block_size=self.block_size, N = self.T)
        elif self.sparse_pattern == 'prev_row':
            lookup_indices, lookup_mask, sparse_indices = previous_row_attention_pattern(block_size=self.block_size, N = self.T)    
        self.lookup_indices, self.lookup_mask, self.sparse_indices = tf.constant(lookup_indices), tf.constant(lookup_mask), tf.constant(sparse_indices)

    def call(self, x):
        if len(x) == 1:
            q = k = v = x[0]
            mask = None
        elif len(x) == 3:
            q = x[0]
            k = x[1]
            v = x[2]
            mask = None
        elif len(x) == 4:
            q = x[0]
            k = x[1]
            v = x[2]
            mask = x[3]

        BS = tf.shape(q)[0]
        NH = tf.shape(q)[1]
        T = tf.shape(q)[2]
        D = tf.shape(q)[3]

        q_ = tf.expand_dims(q,axis=3) #BS,NH,TQ,1,D
        k_lookup = tf.reshape(k,[BS*NH*T,-1])

        n_sparse_indices = tf.shape(self.sparse_indices)[0]
        lookup_indices = tf.tile(tf.cast(self.lookup_indices[tf.newaxis,:,:],tf.int32),[NH,1,1]) + T*tf.range(NH,dtype=tf.int32)[:,tf.newaxis,tf.newaxis]
        lookup_indices = tf.tile(tf.cast(lookup_indices[tf.newaxis,:,:,:],tf.int32),[BS,1,1,1]) + T*NH*tf.range(BS,dtype=tf.int32)[:,tf.newaxis,tf.newaxis,tf.newaxis] #BS,NH,TQ,BL,D

        batch_axis = tf.reshape(tf.ones((BS,n_sparse_indices*NH),dtype=tf.int32)*tf.range(BS,dtype=tf.int32)[:,tf.newaxis],[-1])
        head_axis = tf.tile(tf.reshape(tf.ones((NH,n_sparse_indices),dtype=tf.int32)*tf.range(NH,dtype=tf.int32)[:,tf.newaxis],[-1]),[BS,])
        sparse_indices = tf.tile(tf.cast(self.sparse_indices,tf.int32),[BS*NH,1])
        sparse_indices = tf.concat([batch_axis[:,tf.newaxis],head_axis[:,tf.newaxis],sparse_indices],axis=-1) #BS*NH, 4
        lookup_mask = tf.tile(self.lookup_mask[tf.newaxis,tf.newaxis,:,:],[BS,NH,1,1]) #BS,NH,TQ,BL

        k_ = tf.nn.embedding_lookup(k_lookup,lookup_indices) #BS,NH,TQ,BL,D
        k_ = tf.transpose(k_,perm=[0,1,2,4,3])
        qk = tf.squeeze(tf.linalg.matmul(q_,k_),axis=-2)
        qk_sparse = tf.SparseTensor(indices=tf.cast(sparse_indices,dtype=tf.int64),values=tf.boolean_mask(qk,tf.cast(lookup_mask,dtype=tf.bool)),dense_shape=[BS,NH,T,T]) #BS,NH,TQ,TK
  
        if mask is not None:
            mask = mask[:,tf.newaxis,tf.newaxis,:]
            qk_sparse += (mask * -1e9)

        if self.apply_softmax:
            qk_sparse = tf.sparse.softmax(qk_sparse)

        qkv = batch_sparse_dense_matmul(tf.sparse.reshape(qk_sparse,[BS*NH,T,T]),tf.reshape(v,[BS*NH,T,D]))
        qkv = tf.reshape(qkv,[BS,NH,T,D])

        return qkv

    def compute_output_shape(self,input_shape):
        return input_shape[0]