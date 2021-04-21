import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np

class MSE(tfkl.Layer):
    #x[0]: predicted
    #x[1]: original
    def __init__(self,name=None,lnorm=2,offset=1e-9,normalize=False,trainable=False):
        super(MSE,self).__init__(name=name,trainable=trainable)
        self.offset = offset
        self.normalize = normalize
        self.lnorm = lnorm
    
    def call(self,x):
        mse_error = tf.abs(x[0] - x[1])**self.lnorm
        if self.normalize:
            mse_error = mse_error/(self.offset + tf.abs(x[1])**self.lnorm)
        return mse_error

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'offset': self.offset,
            'normalize': self.normalize,
            'lnorm': self.lnorm
        })
        return config

class Wav2Vec2ContrastiveLoss(tfkl.Layer):
    def __init__(self, negatives_from_different_audios=False, temperature=0.1, negatives_weight=100,name='Wav2Vec2ContrastiveLoss',batch_size=10,seq_len=32,mask_value=-1):
        super(Wav2Vec2ContrastiveLoss, self).__init__(name=name)
        self.negatives_from_different_audios = negatives_from_different_audios
        self.batch_size=batch_size
        self.seq_len=seq_len
        self.temperature=temperature
        self.mask_value=mask_value
        self.negatives_weight=negatives_weight

    def build(self,input_shape):
        input_shape_0 = input_shape[0].as_list()
        mask_different_audios = np.zeros((self.batch_size*input_shape_0[1],self.batch_size*input_shape_0[1]))
        for i in range(self.batch_size):
            mask_different_audios[i*input_shape_0[1]:(i+1)*input_shape_0[1],i*input_shape_0[1]:(i+1)*input_shape_0[1]] = 1
        
        self.mask_different_audios = tf.constant(mask_different_audios,dtype=tf.float32)
        
    def call(self,x):
        #Receives [output1,output2,mask]
        output1 = x[0]
        output2 = x[1]
        mask = tf.reshape(x[2],[1,-1])
        input_shape = tf.shape(x[0])
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        
        output1 = tf.reshape(output1,[-1,output1.shape[-1]])
        output2 = tf.reshape(output2,[-1,output2.shape[-1]])
        
        #sim_ = tf.matmul(output1,tf.transpose(output2))/tf.math.maximum(tf.norm(output1,axis=-1)*tf.norm(output2,axis=-1),1e-8) #cosine similarity
        #sim_ = tf.math.minimum(sim_,1.0) #clippear distancia coseno mayor a 1 (por inestabilidad numerica)
        #sim = tf.math.exp(sim_/self.temperature) #Cuidado de no poner temperatura muy baja ya que puede llevar a overflow

        output1_norm = tf.norm(output1,axis=-1,keepdims=True)
        output2_norm = tf.norm(output2,axis=-1,keepdims=True)
        sim = tf.matmul(output1,tf.transpose(output2))/(tf.matmul(output1_norm,tf.transpose(output2_norm))+1e-9)
        sim = tf.exp(sim/self.temperature)

        mask = tf.cast(mask == self.mask_value,tf.float32)
        tiled_mask = tf.tile(mask,[tf.shape(mask)[-1],1])
        only_masked_against_masked = tf.cast(tiled_mask*tf.transpose(tiled_mask)>0,tf.float32) #Solo uso como negativos los que provienen de regiones enmascaradas
        
        positives_mask = tf.linalg.diag(tf.cast(tf.reshape(mask,[-1]) !=0,tf.float32))
        negatives_mask = only_masked_against_masked - positives_mask #Y solo si la region enmascarada es de otra parte enmascarada
        
        if not self.negatives_from_different_audios:
            negatives_mask = self.mask_different_audios*negatives_mask

        avg_negatives_per_positive = tf.reduce_sum(negatives_mask)/tf.reduce_sum(positives_mask)
        self.add_metric(avg_negatives_per_positive,name='mean_distractors_per_positive')

        num = tf.reduce_sum(sim*positives_mask,axis=0)
        den = self.negatives_weight*tf.reduce_sum(sim*negatives_mask,axis=0)/(tf.reduce_sum(negatives_mask,axis=0)+1e-16)

        closs = tf.reduce_sum(-(tf.math.log(num+1e-16)-tf.math.log(den+1e-16)))/tf.reduce_sum(positives_mask)

        self.add_metric(tf.reduce_mean(num),name='positives_loss')
        self.add_metric(tf.reduce_mean(den/self.negatives_weight),name='negatives_loss')
        self.add_metric(closs,name='contrastive_loss')

        return closs