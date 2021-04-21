import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp

class VQLayer(tfkl.Layer):
    def __init__(self,K,D,beta_commitment = 2.5,mode='quantize',trainable=True,name=None):
        super(VQLayer, self).__init__(name=name)
        e_init = tf.keras.initializers.VarianceScaling(distribution='uniform')
        self.k = K
        self.d = D
        self.beta_commitment = beta_commitment
        self.mode = mode
        self.embeddings = tf.Variable(
                initial_value=e_init(shape=(K, D), dtype="float32"),
                trainable=self.trainable,
            )
  
    def call(self, ze):
        if self.mode == 'decode_indexs':
            return tf.gather(self.embeddings,tf.cast(ze,tf.int32))
        else:
            ze_ = tf.expand_dims(ze,axis=-2) # (batch_size, 1, D)
            distances = tf.norm(self.embeddings-ze_,axis=-1) # (batch_size, K) -> distancia de cada instancia a cada elemento del diccionario
            k = tf.argmin(distances,axis=-1) # indice del elemento con menor distancia
            zq = tf.gather(self.embeddings,k) #elemento del diccionario con menor distancia
            straight_through = tfkl.Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator")([zq,ze]) #Devuelve zq pero propaga a ze

            vq_loss = tf.reduce_mean((tf.stop_gradient(ze) - zq)**2) #Error entre encoder y diccionario propagado al diccionario
            commit_loss = self.beta_commitment*tf.reduce_mean((ze - tf.stop_gradient(zq))**2) #Error entre encoder y diccionario propagado al encoder

            self.add_loss(vq_loss)
            self.add_loss(commit_loss)
            self.add_metric(vq_loss, name='vq_loss')
            self.add_metric(tf.reduce_mean(tf.norm(ze,axis=-1)),name='ze_norm')
            self.add_metric(tf.reduce_mean(tf.norm(zq,axis=-1)),name='zq_norm')
            if self.mode == 'quantize':
                return straight_through
            elif self.mode == 'return_indexs':
                return k

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.k,
            'D': self.d,
            'beta_commitment': self.beta_commitment,
            'mode': self.mode
        })
        return config

class GumbelSoftmaxVQ(tfkl.Layer):
    def __init__(self, codes_per_group, vq_dim, groups=1,temperature=0.5,name=None,affine_group_merge=True,logits_as_input=False,diversity_loss_weight=0.1, diversity_loss='entropy'):
        super(GumbelSoftmaxVQ, self).__init__(name=name)
        self.codes_per_group = codes_per_group
        self.vq_dim = vq_dim
        self.groups = groups
        self.temperature = tf.Variable(temperature,trainable=False)
        self.affine_group_merge = affine_group_merge
        self.logits_as_input = logits_as_input
        self.diversity_loss_type = diversity_loss
        self.diversity_loss_weight = tf.Variable(diversity_loss_weight,trainable=False)

    def build(self,input_shape):
        if not self.logits_as_input:
            self.logits_predictor = tfkl.Dense(units=self.codes_per_group*self.groups, name=self.name+'_logits_estimator')
        self.reshape = tfkl.Reshape(target_shape=(input_shape[1:-1] + [self.groups,self.codes_per_group]))
        self.concatenate = tfkl.Reshape(target_shape=(input_shape[1:-1] + [self.groups*self.vq_dim,]))
        #e_init = tf.keras.initializers.VarianceScaling(distribution='uniform')
        e_init = tf.keras.initializers.RandomUniform(minval=0,maxval=1)
        self.codebook = tf.Variable(
                    initial_value=e_init(shape=(self.groups, self.codes_per_group, self.vq_dim), dtype="float32"),
                    trainable=self.trainable, name=self.name+'_codebook'
                ) #groups codebooks with codes_per_group codes of dimension vq_dim each
        if self.affine_group_merge:
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
            
        self.add_loss(diversity_loss)
        self.add_metric(diversity_loss, name='diversity_loss')
        self.add_metric(self.temperature, name='gumbel_softmax_temperature')
        self.add_metric(self.diversity_loss_weight, name='diversity_loss_weight')

        gumbel_softmax_distribution = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=logits_per_group, 
                                                                        name=self.name+'_gumbel_softmax')
        softmax_samples = gumbel_softmax_distribution.sample() #Sample from gumbel-softmax distribution
        hard_samples = tf.cast(tf.equal(softmax_samples, tf.reduce_max(softmax_samples, axis=-1, keepdims=True)),
                            softmax_samples.dtype) #Turn into hot vector
        hard_samples = tf.stop_gradient(hard_samples - softmax_samples) + softmax_samples #Straight-through estimator
        quantized_x = tf.einsum('...gi,gid->...gd',hard_samples,self.codebook) #Get outputs of the codebook for each instance and group
        #Merge groups
        merged_quantized_x = self.concatenate(quantized_x)
        if self.affine_group_merge:
            return self.merge_groups(merged_quantized_x)
        else:
            return quantized_x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'codes_per_group': self.codes_per_group,
            'vq_dim': self.vq_dim,
            'groups': self.groups,
            'temperature': self.temperature,
            'affine_group_merge': self.affine_group_merge,
            'logits_as_input': self.logits_as_input,
            'diversity_loss_weight': self.diversity_loss_weight
        })
        return config