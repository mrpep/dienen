import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp

class VQLayer(tfkl.Layer):
    def __init__(self,K,D,groups=1,beta_commitment = 2.5,mode='quantize',trainable=True,name=None):
        super(VQLayer, self).__init__(name=name, trainable = trainable)
        self.k = K
        self.d = D
        self.groups = groups
        self.beta_commitment = beta_commitment
        self.mode = mode

    def build(self,input_shape):
        e_init = tf.keras.initializers.VarianceScaling(distribution='uniform')
        self.embeddings = self.add_weight(name='codebook', 
                                         shape=(self.groups, self.k, self.d//self.groups),
                                         dtype="float32",
                                         initializer=e_init,
                                         trainable=self.trainable)
        self.codebook_weight = self.add_weight(name='codebook_weight',
                                               shape=(),
                                               dtype="float32",
                                               initializer=tf.keras.initializers.Ones(),
                                               trainable=False)
        self.residual_weight = self.add_weight(name='residual_weight',
                                               shape=(),
                                               dtype="float32",
                                               initializer=tf.keras.initializers.Zeros(),
                                               trainable=False)

    def call(self, ze):
        if self.mode == 'decode_indexs':
            return tf.gather(self.embeddings,tf.cast(ze,tf.int32))
        else:
            original_shape = tf.shape(ze)
            ze_ = tf.reshape(ze,(-1,self.groups,original_shape[-1]//self.groups))
            
            firstterm = tf.expand_dims(tf.norm(ze_,axis=-1)**2,axis=-1)
            #secondterm = tf.matmul(ze_,tf.transpose(self.embeddings))
            secondterm = tf.einsum('ijk,jlk->ijl',ze_,self.embeddings)
            thirdterm = tf.expand_dims(tf.norm(self.embeddings,axis=-1)**2,axis=0)

            distances = firstterm - 2.0*secondterm + thirdterm
            distances = tf.reshape(distances,tf.concat([original_shape[:-1],distances.shape[-2:]],axis=0))
            
            k = tf.argmin(distances,axis=-1)
            zq = tf.gather(self.embeddings,tf.reshape(k,[-1,self.groups]),axis=1)
            zq = tf.einsum('ijik->ijk', zq)
            zq = tf.transpose(zq,(1,0,2))
            zq = tf.reshape(zq,original_shape)

            straight_through = tfkl.Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator")([zq,ze]) #Devuelve zq pero propaga a ze

            vq_loss = self.codebook_weight*tf.reduce_mean((tf.stop_gradient(ze) - zq)**2)
            commit_loss = self.codebook_weight*self.beta_commitment*tf.reduce_mean((ze - tf.stop_gradient(zq))**2)
            
            self.add_loss(vq_loss)
            self.add_loss(commit_loss)
            self.add_metric(vq_loss, name='vq_loss')
            self.add_metric(tf.reduce_mean(tf.norm(ze,axis=-1)),name='ze_norm')
            self.add_metric(tf.reduce_mean(tf.norm(zq,axis=-1)),name='zq_norm')
            if self.mode == 'quantize':
                return ze*self.residual_weight + self.codebook_weight*straight_through
            elif self.mode == 'return_indexs':
                return k

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.k,
            'D': self.d,
            'groups': self.groups,
            'beta_commitment': self.beta_commitment,
            'mode': self.mode
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_codebook_indices(self,ze):
        original_shape = tf.shape(ze)
        ze_ = tf.reshape(ze,(-1,self.groups,original_shape[-1]//self.groups))
            
        firstterm = tf.expand_dims(tf.norm(ze_,axis=-1)**2,axis=-1)
        secondterm = tf.einsum('ijk,jlk->ijl',ze_,self.embeddings)
        thirdterm = tf.expand_dims(tf.norm(self.embeddings,axis=-1)**2,axis=0)

        distances = firstterm - 2.0*secondterm + thirdterm
        distances = tf.reshape(distances,tf.concat([original_shape[:-1],distances.shape[-2:]],axis=0))
            
        return tf.argmin(distances,axis=-1)

class GumbelSoftmaxVQ(tfkl.Layer):
    def __init__(self, codes_per_group, vq_dim, 
                 groups=1,temperature=0.5,name=None,
                 merge_method='affine',logits_as_input=False,
                 diversity_loss_weight=0.1, diversity_loss='entropy',
                 use_gumbel_noise=True):

        super(GumbelSoftmaxVQ, self).__init__(name=name)
        self.codes_per_group = codes_per_group
        self.vq_dim = vq_dim
        self.groups = groups
        self.temperature = tf.Variable(temperature,trainable=False)
        self.merge_method = merge_method
        self.logits_as_input = logits_as_input
        self.diversity_loss_type = diversity_loss
        self.diversity_loss_weight = tf.Variable(diversity_loss_weight,trainable=False)
        self.use_gumbel_noise = use_gumbel_noise
        self.codebook_weight = tf.Variable(1.0,trainable=False,name='codebook_weight')
        self.residual_weight = tf.Variable(0.0,trainable=False,name='residual_weight')

    def build(self,input_shape):
        if not self.logits_as_input:
            self.logits_predictor = tfkl.Dense(units=self.codes_per_group*self.groups, name=self.name+'_logits_estimator')
        self.reshape = tfkl.Reshape(target_shape=(input_shape[1:-1] + tuple([self.groups,self.codes_per_group])))
        self.concatenate = tfkl.Reshape(target_shape=(input_shape[1:-1] + tuple([self.groups*self.vq_dim,])))
        #e_init = tf.keras.initializers.VarianceScaling(distribution='uniform')
        e_init = tf.keras.initializers.RandomUniform(minval=0,maxval=1)
        self.codebook = tf.Variable(
                    initial_value=e_init(shape=(self.groups, self.codes_per_group, self.vq_dim), dtype="float32"),
                    trainable=self.trainable, name=self.name+'_codebook'
                ) #groups codebooks with codes_per_group codes of dimension vq_dim each
        if self.merge_method == 'affine':
            self.merge_groups = tfkl.Dense(units=self.vq_dim,name=self.name+'_dense_merge_codes')

    def call(self,x):
        if self.logits_as_input:
            logits = x
        else:
            logits = self.logits_predictor(x) #Dense layer to predict logits
        logits_per_group = self.reshape(logits) #Reshape to the groups
        softmax_vals = tf.keras.activations.softmax(logits_per_group, axis=-1)
        #softmax_probs = tf.reduce_mean(tfkl.Reshape((-1,))(),axis=0) #Mean softmax in batch
        hard_code_pred = tf.cast(tf.equal(softmax_vals,tf.reduce_max(softmax_vals,axis=-1,keepdims=True)),tf.float32) #These are the predictions using argmax
        hard_probs = tf.reduce_mean(tf.reshape(hard_code_pred,(-1,self.groups,self.codes_per_group)),axis=0)
        gv = tf.cast(self.groups*self.codes_per_group,tf.float32)
        perplexity = (gv-tf.reduce_sum(tf.exp(-tf.reduce_sum(hard_probs*tf.math.log(hard_probs+1e-7),axis=-1)),axis=-1))/gv
        
        if self.diversity_loss_type == 'entropy':
            diversity_loss = self.diversity_loss_weight*tf.reduce_mean(hard_probs*tf.math.log(hard_probs+1e-12)) #Entropy of that mean softmax
        elif self.diversity_loss_type == 'perplexity':
            diversity_loss = self.diversity_loss_weight*perplexity
        else:
            diversity_loss = None
            
        if diversity_loss is not None:
            self.add_loss(diversity_loss)
            self.add_metric(diversity_loss, name='diversity_loss')
            self.add_metric(self.diversity_loss_weight, name='diversity_loss_weight')

        if self.use_gumbel_noise:
            gumbel_softmax_distribution = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=logits_per_group, 
                                                                        name=self.name+'_gumbel_softmax')
            softmax_samples = gumbel_softmax_distribution.sample() #Sample from gumbel-softmax distribution
            self.add_metric(self.temperature, name='gumbel_softmax_temperature')
        else:
            softmax_samples = logits_per_group

        hard_samples = tf.cast(tf.equal(softmax_samples, tf.reduce_max(softmax_samples, axis=-1, keepdims=True)),
                            softmax_samples.dtype) #Turn into hot vector
        hard_samples = tf.stop_gradient(hard_samples - softmax_samples) + softmax_samples #Straight-through estimator
        quantized_x = tf.einsum('...gi,gid->...gd',hard_samples,self.codebook) #Get outputs of the codebook for each instance and group
        #Merge groups
        merged_quantized_x = self.concatenate(quantized_x)
        if self.merge_method == 'affine':
            return self.residual_weight*x + self.codebook_weight*self.merge_groups(merged_quantized_x)
        elif self.merge_method == 'concatenate':
            return self.residual_weight*x + self.codebook_weight*merged_quantized_x
        elif self.merge_method == 'sum':
            return self.residual_weight*x + self.codebook_weight*tf.reduce_sum(quantized_x,axis=2)
        elif self.merge_method is None:
            return self.residual_weight*x + self.codebook_weight*quantized_x

    def get_codebook_indices(self,x):
        if self.logits_as_input:
            logits = x
        else:
            logits = self.logits_predictor(x) #Dense layer to predict logits
        logits_per_group = self.reshape(logits) #Reshape to the groups
        softmax_vals = tf.keras.activations.softmax(logits_per_group, axis=-1)

        hard_code_pred = tf.cast(tf.equal(softmax_vals,tf.reduce_max(softmax_vals,axis=-1,keepdims=True)),tf.float32) #These are the predictions using argmax
        
        return hard_code_pred

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'codes_per_group': self.codes_per_group,
            'vq_dim': self.vq_dim,
            'groups': self.groups,
            'temperature': self.temperature,
            'merge_method': self.merge_method,
            'logits_as_input': self.logits_as_input,
            'diversity_loss_weight': self.diversity_loss_weight,
            'use_gumbel_noise': self.use_gumbel_noise
        })
        return config

    def compute_output_shape(self,input_shape):
        output_shape = input_shape[:-1] + tuple([self.vq_dim])
        return output_shape