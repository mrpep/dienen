import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params
import numpy as np

class DecayFactorCallback(tf.keras.callbacks.Callback):
    """
    This callback allows to decrease a variable of the model by a factor at each step.
    
    args:
    variable_layer:         layer in which the variable is defined
    variable_name:          name of the variable in that layer
    factor:                 factor by which the variable is multiplied at each step
    """
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

class GradualUnfreezing(tf.keras.callbacks.Callback):
    def __init__(self, epochs=[5,3,3,3,3,3,3,3,5], from_layer=[-2, -5, -6, -7, -8, -9, -10, -11, 0], lr_factor=0.5, initial_lr=0.001):
        super().__init__()
        self.epochs, self.from_layer, self.lr_factor, self.initial_lr = epochs, from_layer, lr_factor, initial_lr
        self.cum_epochs = np.cumsum(epochs)
        self.stage_counter = 0
    
    def unfreeze(self, from_layer, lr, epochs):
        print('GradualUnfreezing: Training from layer {} during {} epochs with lr {}'.format(from_layer,epochs,lr))
        for l in self.model.layers[:from_layer]:
            l.trainable = False
        for l in self.model.layers[from_layer:]:
            l.trainable = True
        
        self.model.compile(loss=self.model.loss, optimizer=self.model.optimizer, metrics=self.model.compiled_metrics._metrics)
        self.model.train_function = self.model.make_train_function()
        tf.keras.backend.set_value(self.model.optimizer.lr,lr)
        print('Trainable weights: {}, Non trainable weights: {}'.format(count_params(self.model.trainable_weights),count_params(self.model.non_trainable_weights)))
        self.current_lr = lr

    def on_train_begin(self,logs=None):
        self.unfreeze(self.from_layer[0], self.initial_lr*self.lr_factor[0], self.epochs[0])

    def on_epoch_end(self,epoch,logs=None):
        if (epoch == self.cum_epochs[self.stage_counter]) & ((self.stage_counter + 1) < len(self.epochs)):
            self.stage_counter += 1
            self.unfreeze(self.from_layer[self.stage_counter], self.current_lr*self.lr_factor[self.stage_counter], self.epochs[self.stage_counter])


