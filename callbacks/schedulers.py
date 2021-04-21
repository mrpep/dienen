import tensorflow as tf

class DecayFactorCallback(tf.keras.callbacks.Callback):
    def __init__(self,variable_layer,variable_name,factor):
        super(DecayFactorCallback, self).__init__()
        self.factor=factor
        self.variable_layer = variable_layer
        self.variable_name = variable_name

    def on_train_begin(self, logs=None):
        variable_layer = [l for l in self.model.layers if l.name == self.variable_layer][0]
        self.updatable_variable = getattr(variable_layer,self.variable_name)
    
    def on_batch_end(self,batch,logs=None):
        old_val = tf.keras.backend.get_value(self.updatable_variable)
        tf.keras.backend.set_value(self.updatable_variable,old_val*self.factor)