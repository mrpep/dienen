import tensorflow as tf
import tensorflow.keras.layers as tfkl

class GELU(tfkl.Layer):
    def __init__(self,name=None,trainable=False,approximate='sigmoid'):
        super(GELU,self).__init__(name=name)
        self.approximate=approximate
    
    def call(self,x):
        if self.approximate == 'tanh':
          return 0.5*x*(1.0 + tf.math.tanh(0.7978845608*(x+0.044715*x**3)))
        elif self.approximate == 'sigmoid':
          return x*tf.math.sigmoid(1.702*x)
        else:
          return x*0.5*(1.0+tf.math.erf(0.70710678118*x))

class ActivityL1L2(tfkl.Layer):
    def __init__(self,l1=0,l2=0,mean=True,name=None,trainable=False):
        super(ActivityL1L2,self).__init__(name=name)
        self.l1 = l1
        self.l2 = l2
        self.mean = mean
    
    def call(self,x):
        if self.mean:
            l1loss=self.l1*tf.reduce_mean(tf.abs(x))
            l2loss=self.l2*tf.reduce_mean(tf.square(x))
        else:
            l1loss=self.l1*tf.reduce_sum(tf.abs(x))
            l2loss=self.l2*tf.reduce_sum(tf.square(x))
        self.add_loss(l1loss+l2loss)
        self.add_metric(l1loss+l2loss,name='{}_l1l2'.format(self.name))
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                      'l1': self.l1,
                      'l2': self.l2,
                      'mean': self.mean
                      })
        return config