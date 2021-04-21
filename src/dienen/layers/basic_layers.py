import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.initializers as tfki

class Abs(tfkl.Layer):
    def __init__(self,name=None, trainable=False):
        super(Abs,self).__init__(name=name)
        
    def call(self,x):
        return tf.math.abs(x)

class Angle(tfkl.Layer):
    def __init__(self,name=None):
        super(Angle,self).__init__(name=name)

    def call(self,x):
        return tf.math.angle(x)

class Argmax(tfkl.Layer):
    def __init__(self,axis=-1,name=None):
        super(ArgMax,self).__init__(name=name)
        self.axis = axis

    def call(self,x):
        return tf.argmax(x,axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis,
        })
        return config

class BatchReshape(tfkl.Layer):
    def __init__(self,target_shape=None,name=None,trainable=False):
        super(BatchReshape, self).__init__(name=name)
        self.target_shape = target_shape

    def call(self,x):
        return tf.reshape(x,self.target_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
        })
        return config

class BooleanLayer(tfkl.Layer):
    def __init__(self,condition=None,value=None,cast='float32',name=None):
        super(BooleanLayer, self).__init__(name=name)
        self.condition = condition
        self.value = value
        self.cast = cast
    
    def call(self,x):
        if self.cast == 'int32':
            self.cast = tf.int32
        elif self.cast == 'float32':
            self.cast = tf.float32
        if self.condition == 'greater':
            return tf.cast(x>self.value,self.cast)
        elif self.condition == 'equal':
            return tf.cast(x==self.value,self.cast)
        else:
            raise Exception('Condition {} not implemented'.format(self.condition))
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'condition': self.condition,
            'value': self.value,
            'cast': self.cast
        })
        return config

class Divide(tfkl.Layer):
    def __init__(self,name=None,trainable=False,offset=1e-9):
        super(Divide,self).__init__(name=name)
        self.offset = offset

    def call(self,x):
        return tf.math.divide(x[0],x[1]+self.offset)

class EmbeddingMask(tfkl.Layer):
    """
    Receives 2 tensors as input: [signal, mask]
    signal: is the tensor we want to modify
    mask: tensor of shape [BatchSize,T] which indicates for each signal which steps we want to replace with another value
    
    masking_values: if None, the vectors corresponding to the elements of the mask != keep_value will be learnt
    mask_shape: shape of the vector to replace in each element of the mask != keep_value
    n_embeddings: number of different vectors
    keep_value: value of the mask which indicates that the input remains unaltered for that step
    """
    def __init__(self,masking_values=None,mask_shape=None,n_embeddings=None,keep_value=-1,name=None):
        super(EmbeddingMask, self).__init__(name=name)
        self.masking_values = masking_values
        self.mask_shape = mask_shape
        self.n_embeddings=n_embeddings
        self.keep_value = keep_value

    def build(self,input_shape):
        print(input_shape[0])
        if self.masking_values is None:
            self.lookup_table = self.add_weight(shape=[self.n_embeddings] + input_shape[0][2:],
                                initializer='normal',
                                trainable=True)

    def call(self,x):
        signal, mask = x
        signal_mask = tf.cast(mask == self.keep_value,tf.float32)
        embeddings = tf.nn.embedding_lookup(self.lookup_table,tf.cast(mask,tf.int32))
        signal_mask = tf.expand_dims(signal_mask,axis=-1)
        return signal*signal_mask + embeddings*(1.0-signal_mask)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'masking_values': self.masking_values,
            'mask_shape': self.mask_shape,
            'n_embeddings': self.n_embeddings,
            'keep_value': self.keep_value
        })
        return config

class ExpandDims(tfkl.Layer):
    def __init__(self,axis=None, name = None, trainable=False):
        super(ExpandDims, self).__init__(name=name)
        self.axis = axis
        
    def call(self, x):
        return tf.expand_dims(x,axis=self.axis)

    def compute_output_shape(self,input_shape):
        from IPython import embed
        output_shape = list(input_shape)
        if self.axis == -1 or self.axis == len(output_shape):
            output_shape.append(1)
        else:
            output_shape.insert(self.axis + 1,1)
        
        return output_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config

class Expression(tfkl.Layer):
    def __init__(self,expr,name=None):
        super(Expression,self).__init__(name=name)
        self.expr = expr
        self.func = lambda x: eval(expr)

    def call(self,x):
        return tfkl.Lambda(self.func)(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'expr': self.expr
        })
        return config

class ExternalNormalize(tfkl.Layer):
    def __init__(self,mean=None,scale=None, name=None, trainable=False):
        super(ExternalNormalize, self).__init__(name=name)
        self.mean = mean
        self.scale = scale

    def call(self,x):
        return (x - self.mean)/self.scale

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mean': self.mean,
            'scale': self.scale
        })
        return config

class ExternalUnnormalize(tfkl.Layer):
    def __init__(self,mean=None,scale=None, name=None, trainable=False):
        super(ExternalUnnormalize, self).__init__(name=name)
        self.mean = mean
        self.scale = scale

    def call(self,x):
        return x*self.scale + self.mean

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mean': self.mean,
            'scale': self.scale
        })
        return config

class Identity(tfkl.Layer):
    def __init__(self,name=None):
        super(Identity,self).__init__(name=name)
    
    def call(self,x):
        return x

class Log(tfkl.Layer):
    def __init__(self,base=None,name=None,offset=1e-9, trainable=False):
        super(Log, self).__init__(name=name)
        self.base = base
        self.offset = offset
        
    def call(self,x):
        if isinstance(x,list):
            x = x[0]
        if self.base:
            numerator = tf.math.log(x+self.offset)
            denominator = tf.math.log(tf.constant(self.base, dtype=numerator.dtype))
            return numerator / denominator
        else:
            return tf.math.log(x+self.offset)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'base': self.base,
            'offset': self.offset
        })
        return config

    def compute_output_shape(self,input_shape):
        return input_shape

class Normalize(tfkl.Layer):
    def __init__(self,normalization_type='mvn',name=None):
        super(Normalize,self).__init__(name=name)
        self.normalization_type = normalization_type

    def call(self,x):
        if self.normalization_type == 'mvn':
            return (x - tf.reduce_mean(x))/tf.math.reduce_std(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'normalization_type': self.normalization_type
        })
        return config

class OneHot(tfkl.Layer):
    def __init__(self,depth=None,on_value=1,off_value=0,axis=-1,name=None):
        super(OneHot,self).__init__(name=name)
        self.depth=depth
        self.on_value=on_value
        self.off_value=off_value
        self.axis=axis

    def call(self,x):
        return tf.one_hot(x,self.depth,on_value=self.on_value,off_value=self.off_value,axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'on_value': self.on_value,
            'off_value': self.off_value,
            'axis': self.axis
        })
        return config
    
class PolarToComplex(tfkl.Layer):
    def __init__(self,name=None):
        super(PolarToComplex,self).__init__(name=name)

    def call(self,x):
        magnitude = tf.cast(x[0],tf.complex128)
        phase = tf.cast(x[1],tf.complex128)

        return magnitude*tf.math.exp(phase*1j)

class ReZeroAdd(tfkl.Layer):
    #From paper ReZero is all you need https://arxiv.org/pdf/2003.04887.pdf
    def __init__(self,name=None):
        super(ReZeroAdd,self).__init__(name=name)
    
    def build(self,input_shape):
        self.alpha = self.add_weight("alpha", shape=[1,], initializer=tfki.Constant(0))

    def call(self,x):
        self.add_metric(self.alpha,name='{}_alpha'.format(self.name))
        return x[0] + self.alpha*x[1]
        
class Slice(tfkl.Layer):
    def __init__(self,begin=None,size=None,name=None,trainable=False):
        super(Slice, self).__init__(name=name)
        self.begin = begin
        self.size = size

    def call(self,x):
        return tf.slice(x,self.begin,self.size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size
        })
        return config

class Split(tfkl.Layer):
  def __init__(self,n_splits,axis,name=None):
    super(Split,self).__init__(name=name)
    self.n_splits = n_splits
    self.axis = axis

  def call(self,x):
    return tf.split(x,self.n_splits,axis=self.axis)

class Squeeze(tfkl.Layer):
    def __init__(self,axis=None, name = None, trainable=False):
        super(Squeeze, self).__init__(name=name)
        self.axis = axis
        
    def call(self, x):
        return tf.squeeze(x,axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config

class Stack(tfkl.Layer):
    def __init__(self,name=None,axis=0):
        super(Stack,self).__init__(name=name)
        self.axis=axis

    def call(self,x):
        return tf.stack(x,axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config

class ToComplex(tfkl.Layer):
    def __init__(self,name=None, trainable=False):
        super(ToComplex,self).__init__(name=name)
        
    def call(self,x):
        return tf.cast(x,tf.complex64)

class TranslateRange(tfkl.Layer):
    def __init__(self,name=None,trainable=False,original_range=None,target_range=None):
        super(TranslateRange,self).__init__(name=name, trainable=trainable)
        self.original_range = original_range
        self.target_range = target_range

    def call(self,x):
        x_center = 0.5*(self.original_range[1] + self.original_range[0])
        y_center = 0.5*(self.target_range[1] + self.target_range[0])
        x_range = self.original_range[1] - self.original_range[0]
        y_range = self.target_range[1] - self.target_range[0]

        return (x - x_center)*y_range/x_range + y_center

    def compute_output_shape(self,input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'original_range': self.original_range,
            'target_range': self.target_range
        })
        return config

class WeightedAverage(tf.keras.layers.Layer):
    def __init__(self, weights='trainable',axis=1, initializer = 'ones', name=None):
        super(WeightedAverage, self).__init__(name=name)
        self.avg_weights = weights
        self.axis = axis
        self.initializer = initializer

    def build(self, input_shape):
        weights_shape = np.ones_like(input_shape).astype(int)
        n_weights = input_shape[self.axis]
        weights_shape[-1] = n_weights
        
        if self.avg_weights == 'trainable':
            if self.initializer == 'zeros':
                initializer = tfki.Constant(0)
            elif self.initializer == 'ones':
                initializer = tfki.Constant(1)
            self.kernel = self.add_weight("avg_weights", shape=weights_shape, initializer=initializer)
        else:
            self.kernel = np.ones(weights_shape)
            self.kernel[:] = self.avg_weights

        self.perm = list(range(len(weights_shape)))
        self.perm[self.axis] = self.perm[-2]
        self.perm[-2] = self.axis

    def call(self, input):
        weights = tf.abs(self.kernel)
        res = tf.matmul(weights/tf.reduce_sum(weights),tf.transpose(input,self.perm))
        return tf.transpose(res,self.perm)