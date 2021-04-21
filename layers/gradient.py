import tensorflow.keras.layers as tfkl
import tensorflow as tf

class GradientMultiplier(tfkl.Layer):
    def __init__(self,multiplier=1.0,name=None):
        super(GradientMultiplier,self).__init__(name=name)
        self.multiplier = multiplier
        @tf.custom_gradient
        def custom_op(x):
            def custom_grad(dy):
                grad = dy*self.multiplier
                return grad
            return x, custom_grad
        self.custom_op = custom_op

    def call(self,x):
        return self.custom_op(x)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'multiplier': self.multiplier
        })
        return config