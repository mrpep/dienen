import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.initializers as tfki
from .convolutions import DynamicConvolution, GLU
from .basic_layers import Split, ReZeroAdd
from .initialization import InitializerScaler, InitializeWith
from .sparse_attention import BlockSparseProductAttention

class MultiheadAttention(tfkl.Layer):
    def __init__(self,name=None,
                 d_model=512, 
                 d_proj=64, 
                 n_heads=8, 
                 use_bias = False,
                 qkvw_init_scale = [1,1,1,1],
                 apply_softmax=True,
                 sparse_pattern=None,
                 sparse_block_size=16,
                 trainable=True,
                 relative_attention_type=False,
                 shape_2d=None,
                 share_pe_heads=True,
                 pe_initialization='uniform',
                 cls_token=True,
                 q_initializer='uniform',
                 k_initializer='uniform',
                 v_initializer='uniform',
                 o_initializer='uniform'):

        super(MultiheadAttention,self).__init__(name=name)
        self.d_model = d_model
        self.d_proj = d_proj
        self.n_heads = n_heads
        self.sparse_pattern = sparse_pattern

        if sparse_pattern is None:
            self.scaled_dot_product_attention = ScaledDotProductAttention(name = self.name + '_attention', apply_softmax = apply_softmax, relative_attention_type=relative_attention_type,
                 shape_2d=shape_2d, share_pe_heads=share_pe_heads, pe_initialization=pe_initialization, cls_token = cls_token) 
        else:
            self.scaled_dot_product_attention = BlockSparseProductAttention(name = self.name + '_block_sparse_attention', block_size=sparse_block_size,sparse_pattern=sparse_pattern,apply_softmax = apply_softmax)
        
        if q_initializer == 'uniform':
            q_initializer = tfki.GlorotUniform()
        elif q_initializer == 'zeros':
            q_initializer = tfki.Zeros()
        elif q_initializer == 'identity':
            q_initializer = tfki.Identity()
        
        if k_initializer == 'uniform':
            k_initializer = tfki.GlorotUniform()
        elif k_initializer == 'zeros':
            k_initializer = tfki.Zeros()
        elif k_initializer == 'identity':
            k_initializer = tfki.Identity()

        if v_initializer == 'uniform':
            v_initializer = tfki.GlorotUniform()
        elif v_initializer == 'zeros':
            v_initializer = tfki.Zeros()
        elif v_initializer == 'identity':
            v_initializer = tfki.Identity()

        if o_initializer == 'uniform':
            o_initializer = tfki.GlorotUniform()
        elif o_initializer == 'zeros':
            o_initializer = tfki.Zeros()
        elif o_initializer == 'identity':
            o_initializer = tfki.Identity()

        self.wq = tf.keras.layers.Dense(d_proj*n_heads,use_bias=use_bias,name=self.name + '_wq',kernel_initializer=InitializerScaler(q_initializer,qkvw_init_scale[0]), trainable=trainable)
        self.wk = tf.keras.layers.Dense(d_proj*n_heads,use_bias=use_bias,name=self.name + '_wk',kernel_initializer=InitializerScaler(k_initializer,qkvw_init_scale[1]), trainable=trainable)
        self.wv = tf.keras.layers.Dense(d_proj*n_heads,use_bias=use_bias,name=self.name + '_wv',kernel_initializer=InitializerScaler(v_initializer,qkvw_init_scale[2]), trainable=trainable)
        self.wo = tf.keras.layers.Dense(d_model,use_bias=use_bias,name=self.name + '_wo',kernel_initializer=InitializerScaler(o_initializer,qkvw_init_scale[3]), trainable=trainable)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.d_proj))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def build(self,input_shape):
        if self.sparse_pattern is not None:
            self.scaled_dot_product_attention.T = self.T

    def call(self,q,k,v,mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_proj*n_heads)
        k = self.wk(k)  # (batch_size, seq_len, d_proj*n_heads)
        v = self.wv(v)  # (batch_size, seq_len, d_proj*n_heads)

        q = self.split_heads(q, batch_size)  # (batch_size, n_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, n_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, n_heads, seq_len_v, depth)
        if mask is not None:
            scaled_attention = self.scaled_dot_product_attention([q, k, v, mask])
        else:
            scaled_attention = self.scaled_dot_product_attention([q, k, v])
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_proj*self.n_heads))  # (batch_size, seq_len_q, d_model)

        output = self.wo(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output

class PositionalEmbeddingLookup(tfkl.Layer):
    def __init__(self, name=None,trainable=True, initialization='normal',repeat=1,tile=1,cls_token=True):
        super(PositionalEmbeddingLookup, self).__init__(name=name,trainable=True)
        self.initialization = initialization
        self.supports_masking = True
        self.tile, self.repeat, self.cls_token = tile, repeat, cls_token

    def build(self, input_shape):
        T = input_shape[1]
        if self.tile>1:
            if self.cls_token:
                T = (T-1)//self.tile + 1
            else:
                T = T//self.tile
        if self.repeat>1:
            if self.cls_token:
                T = (T-1)//self.repeat + 1
            else:
                T = T//self.repeat            

        self.lookup_table = self.add_weight(shape=(T,input_shape[-1]),
                               initializer=self.initialization,
                               trainable=True)

        self.positions = tf.constant(np.arange(0,input_shape[1]))

    def call(self, x):
        if self.tile > 1:
            if self.cls_token:
                lookup_table = tf.concat([tf.expand_dims(self.lookup_table[0,:],axis=0), tf.tile(self.lookup_table[1:,:], (self.tile,1))],axis=0)
            else:
                lookup_table = tf.tile(self.lookup_table, (1,self.tile,1))
            return x + tf.nn.embedding_lookup(lookup_table,self.positions)
        if self.repeat > 1:
            if self.cls_token:
                lookup_table = tf.concat([tf.expand_dims(self.lookup_table[0,:],axis=0), tf.repeat(self.lookup_table[1:,:], self.repeat, axis=0)],axis=0)
            else:
                lookup_table = tf.repeat(self.lookup_table, self.repeat,axis=0)
            return x + tf.nn.embedding_lookup(lookup_table,self.positions)
        return x + tf.nn.embedding_lookup(self.lookup_table,self.positions)
        

class PositionalEncoding(tfkl.Layer):
    def __init__(self,name=None,scale=1.0,trainable=False):
        super(PositionalEncoding,self).__init__(name=name, trainable=trainable)
        self.scale = scale
        self.supports_masking = True

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x, mask=None):
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
    def __init__(self,apply_softmax=True,name=None,trainable=False, relative_attention_type=False,
                 shape_2d=None, share_pe_heads=True, pe_initialization='uniform',cls_token=True):
        super(ScaledDotProductAttention,self).__init__(name=name)
        self.apply_softmax = apply_softmax
        self.relative_attention_type = relative_attention_type
        self.shape_2d = shape_2d
        self.share_pe_heads = share_pe_heads
        self.pe_initialization = pe_initialization
        self.cls_token = cls_token

    def get_2dindices_from_sequence(self,w,h):
        h_indices = np.tile(np.arange(h),w)
        w_indices = np.reshape(np.arange(0,w)[:,np.newaxis]*np.ones((w,h)),-1)

        return np.stack([w_indices, h_indices]).T

    def get_idx_distances(self,w,h,cls_token=True,offset=True,cls_token_correction=False):
        idxs = self.get_2dindices_from_sequence(w,h)
        idxs_tiled = np.tile(idxs[:,:,np.newaxis],(1,1,idxs.shape[0]))
        idxs_tiled = np.transpose(idxs_tiled,(0,2,1))
        r_w = idxs_tiled[:,:,0].T - idxs_tiled[:,:,0]
        r_h = idxs_tiled[:,:,1].T - idxs_tiled[:,:,1]
        if cls_token:
            r_w = np.concatenate([np.zeros((1,w*h)),r_w],axis=0)
            r_w = np.concatenate([np.zeros((w*h+1,1)),r_w],axis=1)
            r_h = np.concatenate([np.zeros((1,w*h)),r_h],axis=0)
            r_h = np.concatenate([np.zeros((w*h+1,1)),r_h],axis=1)
        if offset:
            r_w = r_w - r_w.min()
            r_h = r_h - r_h.min()
        
        #CLS token correction (cls can see everything but other tokens cant see CLS, making it harder to learn long distance relations)
        if cls_token_correction:
            r_w[1:,0] = -1e9

        return idxs,r_w, r_h

    def get_idxs_matrix(self, n_heads, cls_token=True):
        idxs, rw, rh = self.get_idx_distances(self.shape_2d[0],self.shape_2d[1],cls_token)
        idxs_for_rw = []
        idxs_for_rh = []
        for h in range(n_heads):
            for i in range(rw.shape[0]):
                for j in range(rw.shape[1]):
                    idxs_for_rw.append([h,i,rw[i,j]])
                    idxs_for_rh.append([h,i,rh[i,j]])
        rw_idxs_tf = tf.constant(np.array(idxs_for_rw),dtype=tf.int32)
        rh_idxs_tf = tf.constant(np.array(idxs_for_rh),dtype=tf.int32)

        return rw_idxs_tf, rh_idxs_tf

    def build(self,input_shape):
        input_shape = input_shape[0]
        self.n_heads = input_shape[1]
        self.d_proj = input_shape[-1]

        if self.relative_attention_type == 'huang2d':
            self.rw_idxs, self.rh_idxs = self.get_idxs_matrix(self.n_heads,self.cls_token)
            if self.share_pe_heads:
                n_heads_trainable = 1
            else:
                n_heads_trainable = self.n_heads

            self.er_w = self.add_weight(shape=[n_heads_trainable,self.shape_2d[0]*2-1,self.d_proj],
                               initializer=self.pe_initialization,
                               trainable=True,name='rpe_w')
            self.er_h = self.add_weight(shape=[n_heads_trainable,self.shape_2d[1]*2-1,self.d_proj],
                               initializer=self.pe_initialization,
                               trainable=True,name='rpe_h')

        elif self.relative_attention_type == 'huang1d':
            rw_idxs = np.arange(0,input_shape[1],dtype=np.int32)
            rw_idxs = np.tile(rw_idxs[:,np.newaxis],(1,input_shape[1]))
            rw_idxs = rw_idxs.T - rw_idxs
            rw_idxs = rw_idxs - np.min(rw_idxs)

            idxs_for_rw = []
            for h in range(self.n_heads):
                for i in range(rw_idxs.shape[0]):
                    for j in range(rw_idxs.shape[1]):
                        idxs_for_rw.append([h,i,rw_idxs[i,j]])
            self.rw_idxs_tf = tf.constant(np.array(idxs_for_rw),dtype=tf.int32)

            if self.share_pe_heads:
                n_heads_trainable = 1
            else:
                n_heads_trainable = self.n_heads
            self.er = self.add_weight(shape=[n_heads_trainable,input_shape[1]*2-1,self.d_proj],
                                        initializer=self.pe_initialization,
                                        trainable=True,name='rpe')
            

        elif self.relative_attention_type == 'alibi2d':
            idxs, rw, rh = self.get_idx_distances(self.shape_2d[0],self.shape_2d[1],self.cls_token,offset=False,cls_token_correction=True)
            ms = 2**(-np.linspace(1,8,self.n_heads//2))
            rw_alibi = -np.abs(rw)
            rh_alibi = -np.abs(rh)
            rh_alibi = np.tile(rh_alibi,(self.n_heads//2,1,1))
            rh_alibi_heads = ms[:,np.newaxis,np.newaxis]*rh_alibi
            rw_alibi = np.tile(rw_alibi,(self.n_heads//2,1,1))
            rw_alibi_heads = ms[:,np.newaxis,np.newaxis]*rw_alibi
            all_heads = np.concatenate([rw_alibi_heads,rh_alibi_heads],axis=0)
            self.r_matrix = self.add_weight(shape=[self.n_heads,all_heads.shape[1],all_heads.shape[2]],
                                            initializer=InitializeWith(tf.constant(all_heads)),
                                            trainable=False,
                                            name='r_matrix')
        elif self.relative_attention_type == 'alibi2d_timeonly':
            idxs, rw, rh = self.get_idx_distances(self.shape_2d[0],self.shape_2d[1],self.cls_token,offset=False,cls_token_correction=True)
            ms = 2**(-np.linspace(1,8,self.n_heads))
            rw_alibi = -np.abs(rw)
            rw_alibi = np.tile(rw_alibi,(self.n_heads,1,1))
            rw_alibi_heads = ms[:,np.newaxis,np.newaxis]*rw_alibi
            self.r_matrix = self.add_weight(shape=[self.n_heads,rw_alibi_heads.shape[1],rw_alibi_heads.shape[2]],
                                            initializer=InitializeWith(tf.constant(rw_alibi_heads)),
                                            trainable=False,
                                            name='r_matrix')            

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

        if self.relative_attention_type == 'huang2d':
            if self.share_pe_heads:
                er_w = tf.tile(self.er_w,(self.n_heads,1,1))
                er_h = tf.tile(self.er_h,(self.n_heads,1,1))
            else:
                er_w = self.er_w
                er_h = self.er_h
            bs = tf.shape(q)[0]
            bs_idx = tf.repeat(tf.range(0,bs),self.rw_idxs.shape[0])
            rw_idxs = tf.concat([bs_idx[:,tf.newaxis],tf.tile(self.rw_idxs,(bs,1))],axis=1)
            rh_idxs = tf.concat([bs_idx[:,tf.newaxis],tf.tile(self.rh_idxs,(bs,1))],axis=1)
            qerw = tf.matmul(q,tf.expand_dims(er_w,axis=0),transpose_b=True)
            qerh = tf.matmul(q,tf.expand_dims(er_h,axis=0),transpose_b=True)
            sw = tf.gather_nd(qerw,rw_idxs)
            sw = tf.reshape(sw,tf.shape(matmul_qk))
            sh = tf.gather_nd(qerh,rh_idxs)
            sh = tf.reshape(sh,tf.shape(matmul_qk))
            matmul_qk = matmul_qk + sw + sh
        elif self.relative_attention_type == 'huang1d':
            if self.share_pe_heads:
                er = tf.tile(self.er,(self.n_heads,1,1))
            else:
                er = self.er
            bs = tf.shape(q)[0]
            bs_idx = tf.repeat(tf.range(0,bs),self.rw_idxs_tf.shape[0])
            rw_idxs = tf.concat([bs_idx[:,tf.newaxis],tf.tile(self.rw_idxs_tf,(bs,1))],axis=1)
            qerw = tf.matmul(q,tf.expand_dims(er,axis=0),transpose_b=True)
            sw = tf.gather_nd(qerw,rw_idxs)
            sw = tf.reshape(sw,tf.shape(matmul_qk))
            matmul_qk = matmul_qk + sw         
        elif self.relative_attention_type == 'alibi2d' or self.relative_attention_type == 'alibi2d_timeonly':
            matmul_qk = matmul_qk + self.r_matrix

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            mask = mask[:,tf.newaxis,:]
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
                rezero=False,
                sparse_pattern=None,
                sparse_block_size=16,
                relative_attention_type=False,
                shape_2d=None, 
                share_pe_heads=True, 
                pe_initialization='uniform',
                cls_token=True,
                trainable=True,
                q_initializer='uniform',
                k_initializer='uniform',
                v_initializer='uniform',
                o_initializer='uniform'):

        super(TransformerBlock,self).__init__(name=name, trainable = trainable)
        self.rezero = rezero
        if self.rezero:
            self.rezero1 = ReZeroAdd(trainable=trainable)
            self.rezero2 = ReZeroAdd(trainable=trainable)

        if not isinstance(qkvw_init_scale,list):
            qkvw_init_scale = [qkvw_init_scale]*4
        
        self.masking = tfkl.Masking(trainable=trainable)
        self.sparse_pattern = sparse_pattern

        self.mha = MultiheadAttention(name = self.name + '_mha',
                                        d_model=d_model, 
                                        d_proj=d_proj, 
                                        n_heads=n_heads, 
                                        use_bias = use_bias,
                                        qkvw_init_scale = qkvw_init_scale,
                                        apply_softmax=apply_softmax,
                                        sparse_pattern=sparse_pattern,
                                        sparse_block_size=sparse_block_size,
                                        trainable=trainable,
                                        relative_attention_type=relative_attention_type,
                                        shape_2d=shape_2d, 
                                        share_pe_heads=share_pe_heads, 
                                        pe_initialization=pe_initialization,
                                        cls_token=cls_token,
                                        q_initializer=q_initializer,
                                        k_initializer=k_initializer,
                                        v_initializer=v_initializer,
                                        o_initializer=o_initializer
                                        )

        self.dropout_1 = tfkl.Dropout(residual_dropout, name = self.name + '_dropout_1', trainable=trainable)
        self.dropout_2 = tfkl.Dropout(residual_dropout, name = self.name + '_dropout_2', trainable=trainable)
        self.ln1 = tfkl.LayerNormalization(epsilon=1e-6, name = self.name + '_layernorm_1', trainable=trainable)
        self.ln2 = tfkl.LayerNormalization(epsilon=1e-6, name = self.name + '_layernorm_2', trainable=trainable)

        self.ff1 = tfkl.Conv1D(ff_dim,conv_kernel_size,activation='relu',name=self.name + '_ff_1', use_bias = use_bias,padding='SAME', trainable=trainable)
        self.ff2 = tfkl.Conv1D(d_model,conv_kernel_size,name=self.name + '_ff_2', use_bias = use_bias,padding='SAME', trainable=trainable)

        self.prenorm = pre_normalization

        self.supports_masking = True

    def build(self,input_shape):
        if self.sparse_pattern is not None:
            self.mha.T = input_shape[1]

    def call(self,x,training,mask=None):
        if mask is not None:
            x = self.masking(x)
            mask = 1.0 - tf.cast(mask,tf.float32)
        ln1_out = self.ln1(x)
        if mask is not None:
            mha = self.mha(ln1_out,ln1_out,ln1_out,mask)
        else:
            mha = self.mha(ln1_out,ln1_out,ln1_out)

        out1 = x + mha
        ln2_out = self.ln2(out1)
        ff = self.ff1(ln2_out)
        ff = self.dropout_1(ff)
        ff = self.ff2(ff)
        ff = self.dropout_2(ff)
        return ff + out1

        #out1 = self.dropout_1(mha,training=training)
        #if self.prenorm:
        #    out1 = self.ln1(out1)
        #    if self.rezero:
        #        out1 = self.rezero1([x,out1])
        #    else:
        #        out1 = out1 + x
        #else:
        #    if self.rezero:
        #        out1 = self.rezero1([x,out1])
        #    else:
        #        out1 = out1 + x
        #    out1 = self.ln1(out1)
        #ff = self.ff1(out1)
        #ff = self.ff2(ff)
        #out2 = self.dropout_2(ff,training=training)
        #if self.prenorm:
        #    out2 = self.ln2(out2)
        #    if self.rezero:
        #        out2 = self.rezero2([out1,out2])
        #    else:
        #        out2 = out2 + out1
        #else:
        #    if self.rezero:
        #        out2 = self.rezero2([out1,out2])
        #    else:
        #        out2 = out2 + out1
        #    out2 = self.ln2(out2)
        #return out2

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

