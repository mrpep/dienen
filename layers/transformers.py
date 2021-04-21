import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.initializers as tfki
from .convolutions import DynamicConvolution, GLU
from .basic_layers import Split, ReZeroAdd
from .initialization import InitializerScaler

class MultiheadAttention(tfkl.Layer):
    def __init__(self,name=None,
                 d_model=512, 
                 d_proj=64, 
                 n_heads=8, 
                 use_bias = False,
                 qkvw_init_scale = [1,1,1,1],
                 apply_softmax=True):
        super(MultiheadAttention,self).__init__(name=name)
        self.d_model = d_model
        self.d_proj = d_proj
        self.n_heads = n_heads

        self.scaled_dot_product_attention = ScaledDotProductAttention(name = self.name + '_attention', apply_softmax = apply_softmax) 

        self.wq = tf.keras.layers.Dense(d_proj*n_heads,use_bias=use_bias,name=self.name + '_wq',kernel_initializer=InitializerScaler(tfki.GlorotUniform(),qkvw_init_scale[0]))
        self.wk = tf.keras.layers.Dense(d_proj*n_heads,use_bias=use_bias,name=self.name + '_wk',kernel_initializer=InitializerScaler(tfki.GlorotUniform(),qkvw_init_scale[1]))
        self.wv = tf.keras.layers.Dense(d_proj*n_heads,use_bias=use_bias,name=self.name + '_wv',kernel_initializer=InitializerScaler(tfki.GlorotUniform(),qkvw_init_scale[2]))
    
        self.wo = tf.keras.layers.Dense(d_model,use_bias=use_bias,name=self.name + '_wo',kernel_initializer=InitializerScaler(tfki.GlorotUniform(),qkvw_init_scale[3]))

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.d_proj))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,q,k,v,mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_proj*n_heads)
        k = self.wk(k)  # (batch_size, seq_len, d_proj*n_heads)
        v = self.wv(v)  # (batch_size, seq_len, d_proj*n_heads)

        q = self.split_heads(q, batch_size)  # (batch_size, n_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, n_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, n_heads, seq_len_v, depth)

        scaled_attention = self.scaled_dot_product_attention([q, k, v, mask])
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_proj*self.n_heads))  # (batch_size, seq_len_q, d_model)

        output = self.wo(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output

class PositionalEmbeddingLookup(tfkl.Layer):
    def __init__(self, name=None,trainable=True, initialization='normal'):
        super(PositionalEmbeddingLookup, self).__init__(name=name,trainable=True)
        self.initialization = initialization

    def build(self, input_shape):
        self.lookup_table = self.add_weight(shape=(input_shape[1:]),
                               initializer=self.initialization,
                               trainable=True)
        self.positions = tf.constant(np.arange(0,input_shape[1]))
    def call(self, x):
        return x + tf.nn.embedding_lookup(self.lookup_table,self.positions)

class PositionalEncoding(tfkl.Layer):
    def __init__(self,name=None,scale=1.0):
        super(PositionalEncoding,self).__init__(name=name)
        self.scale = scale

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x):
        position = x.shape[-2]
        d_model = x.shape[-1]

        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
      
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
          
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return x + self.scale*tf.cast(pos_encoding, dtype=tf.float32)

class ScaledDotProductAttention(tfkl.Layer):
    def __init__(self,apply_softmax=True,name=None,trainable=False):
        super(ScaledDotProductAttention,self).__init__(name=name)
        self.apply_softmax = apply_softmax

    def call(self,inputs):
        if len(inputs) == 1:
            q = k = v = inputs[0]
            mask = None
        elif len(inputs) == 3:
            q = inputs[0]
            k = inputs[1]
            v = inputs[2]
            mask = None
        elif len(inputs) == 4:
            q = inputs[0]
            k = inputs[1]
            v = inputs[2]
            mask = inputs[3]

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        if self.apply_softmax:
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        else:
            attention_weights = scaled_attention_logits

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output

class TransformerBlock(tfkl.Layer):
    def __init__(self,
                name=None,
                d_model=512, 
                d_proj=64, 
                n_heads=8, 
                residual_dropout=0.1, 
                ff_dim=2048, 
                use_bias = False, 
                conv_kernel_size = 1, 
                pre_normalization=False,
                qkvw_init_scale=1, 
                apply_softmax=True,
                rezero=False):

        super(TransformerBlock,self).__init__(name=name)
        self.rezero = rezero
        if self.rezero:
            self.rezero1 = ReZeroAdd()
            self.rezero2 = ReZeroAdd()
        self.mha = MultiheadAttention(name = self.name + '_mha',
                                      d_model=d_model, 
                                      d_proj=d_proj, 
                                      n_heads=n_heads, 
                                      use_bias = use_bias,
                                      qkvw_init_scale = qkvw_init_scale,
                                      apply_softmax=apply_softmax)
        self.dropout_1 = tfkl.Dropout(residual_dropout, name = self.name + '_dropout_1')
        self.dropout_2 = tfkl.Dropout(residual_dropout, name = self.name + '_dropout_2')
        self.ln1 = tfkl.LayerNormalization(epsilon=1e-6, name = self.name + '_layernorm_1')
        self.ln2 = tfkl.LayerNormalization(epsilon=1e-6, name = self.name + '_layernorm_2')

        self.ff1 = tfkl.Conv1D(ff_dim,conv_kernel_size,activation='relu',name=self.name + '_ff_1', use_bias = use_bias,padding='SAME')
        self.ff2 = tfkl.Conv1D(d_model,conv_kernel_size,name=self.name + '_ff_2', use_bias = use_bias,padding='SAME')

        self.prenorm = pre_normalization

    def call(self,x,training,mask=None):
        mha = self.mha(x,x,x,mask)
        out1 = self.dropout_1(mha,training=training)
        if self.prenorm:
            out1 = self.ln1(out1)
            if self.rezero:
                out1 = self.rezero1([x,out1])
            else:
                out1 = out1 + x
        else:
            if self.rezero:
                out1 = self.rezero1([x,out1])
            else:
                out1 = out1 + x
            out1 = self.ln1(out1)
        ff = self.ff1(out1)
        ff = self.ff2(ff)
        out2 = self.dropout_2(ff,training=training)
        if self.prenorm:
            out2 = self.ln2(out2)
            if self.rezero:
                out2 = self.rezero2([out1,out2])
            else:
                out2 = out2 + out1
        else:
            if self.rezero:
                out2 = self.rezero2([out1,out2])
            else:
                out2 = out2 + out1
            out2 = self.ln2(out2)
        return out2

class TransformerLiteBlock(tfkl.Layer):
    def __init__(self,name=None,d_model=512, n_heads=8, glu_kernel_size=4,dynamic_kernel_size=4, dynamic_H=32, projection_kernel_size=1,ffn_kernel_size=1,residual_dropout=0.1, use_bias = False, conv_kernel_size = 1, pre_normalization=False, apply_softmax=True):
        super(TransformerLiteBlock,self).__init__(name=name)

        self.projection_layer = tfkl.Conv1D(d_model,projection_kernel_size)
        self.split_layer = Split(n_splits=2,axis=-1) #Splits input channels in 2 (one for each branch)

        self.mha = MultiheadAttention(name = self.name + '_mha',d_model=d_model, d_proj=d_model, n_heads=n_heads, use_bias = use_bias, apply_softmax=apply_softmax)

        self.glu = GLU(d_model//2,glu_kernel_size)
        self.dynamic_conv = DynamicConvolution(dynamic_kernel_size,dynamic_H)
        self.cnn_fc = tfkl.Conv1D(d_model//2,ffn_kernel_size)

        self.concat = tfkl.Concatenate()
        self.unprojection_layer = tfkl.Conv1D(d_model,projection_kernel_size)

        self.dropout_1 = tfkl.Dropout(residual_dropout, name = self.name + '_dropout_1')
        self.dropout_2 = tfkl.Dropout(residual_dropout, name = self.name + '_dropout_2')
        self.ln1 = tfkl.LayerNormalization(epsilon=1e-6, name = self.name + '_layernorm_1')

        self.ff1 = tfkl.Conv1D(d_model,conv_kernel_size,activation='relu',name=self.name + '_ff_1', use_bias = use_bias,padding='SAME')

        self.prenorm = pre_normalization

    def call(self,x,training,mask=None):
        projected = self.projection_layer(x)
        split_mha, split_conv = self.split_layer(projected)
        mha = self.mha(split_mha,split_mha,split_mha,mask)

        dyn = self.glu(split_conv)
        dyn = self.dynamic_conv(dyn)
        dyn = self.cnn_fc(dyn)

        lsra = self.concat([mha,dyn])
        lsra = self.unprojection_layer(lsra)

        out1 = self.dropout_1(lsra,training=training)
        if self.prenorm:
            out1 = self.ln1(out1)
            out1 = out1 + x
        else:
            out1 = out1 + x
            out1 = self.ln1(out1)
        ff = self.ff1(out1)
        return ff

