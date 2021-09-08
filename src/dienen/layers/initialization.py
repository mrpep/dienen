import tensorflow.keras.initializers as tfki
import tensorflow as tf

class InitializerScaler(tfki.Initializer):
    def __init__(self, initializer, scale):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        return self.scale*self.initializer(shape, dtype=None, **kwargs)

    def get_config(self):  # To support serialization
        return {"scale": self.scale, "initializer": self.initializer}

class InitializeWith(tfki.Initializer):
    def __init__(self,constant_tensor):
        self.constant_tensor = constant_tensor
  
    def __call__(self,shape,dtype=None,**kwargs):
        return tf.cast(self.constant_tensor,dtype)
      
    def get_config(self):
        return {'constant_tensor': self.constant_tensor}