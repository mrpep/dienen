from .node import DienenNode
from dienen.utils import get_classes_in_module, get_class_signature
import tensorflow.keras.layers as tfkl
import tensorflow_addons as tfa
import dienen.layers
import tensorflow.keras.initializers as initializers

class Layer(DienenNode):
    def __init__(self):
        self.valid_nodes = ['name','input','time_distributed','trainable','bidirectional','class','normalize_weights','kernel_initializer_args','mask']
        self.required_nodes = []

    def set_layer_modules(self,layer_modules):
        if not isinstance(layer_modules,list):
            layer_modules = [layer_modules]
        self.layer_modules = layer_modules

    def set_config(self,config):
        self.layer_type = config['class']
        if self.layer_type == 'Input':
            #self.required_nodes.append('input_shape')
            self.valid_nodes = self.valid_nodes + ['input_shape','batch_shape']
        
        available_layers = {}
        for module in self.layer_modules:
            available_layers.update(get_classes_in_module(module)) #list of layers in backend

        self.layer = available_layers[self.layer_type] #instantiate layer from backend

        layer_available_args, layer_available_kwargs = get_class_signature(self.layer) #get its args and kwargs
        layer_available_kwargs = layer_available_kwargs + ['name','trainable'] #add name to kwargs (sometimes missing)
        if self.layer_type == 'Input':
            layer_available_kwargs = layer_available_kwargs + ['batch_shape']
        
        self.valid_nodes = self.valid_nodes + layer_available_kwargs + layer_available_args 
        self.required_nodes = self.required_nodes + layer_available_args

        super().set_config(config)

        self.name = self.config.get('name',None)
        self.input = self.config.get('input',None)
        self.mask = self.config.get('mask',None)

        kernel_initializer_args = self.config.get('kernel_initializer_args',None)

        if kernel_initializer_args is not None:
            available_initializers = get_classes_in_module(initializers)
            initializer = available_initializers[self.config.get('kernel_initializer',None)](**kernel_initializer_args)
            self.config.pop('kernel_initializer_args')
            self.config['kernel_initializer'] = initializer

        #Check for args and kwargs in config that can be passed to layer
        self.args_to_pass = [self.config[k] for k in layer_available_args]
        self.kwargs_to_pass = {k:self.config[k] for k in list(self.config.keys()) if k in layer_available_kwargs}
        
        #Layer modifiers (add batch normalization, dropout or make it time distributed)
        self.time_distributed = self.config.get('time_distributed',False)
        self.bidirectional = self.config.get('bidirectional',False)
        self.weight_normalization = self.config.get('normalize_weights',False)

    def create(self):
        #Call layer constructor
        if self.time_distributed or self.bidirectional:
            if self.time_distributed:
                self.layer = tfkl.TimeDistributed(self.layer(*self.args_to_pass,**self.kwargs_to_pass))
                self.layer._name=self.name
            if self.bidirectional:
                self.layer = tfkl.Bidirectional(self.layer(*self.args_to_pass,**self.kwargs_to_pass))
                self.layer._name=self.name
            if self.weight_normalization:
                self.layer = tfa.layers.WeightNormalization(self.layer(*self.args_to_pass,**self.kwargs_to_pass))
        else:
            self.layer = self.layer(*self.args_to_pass,**self.kwargs_to_pass)

        return self.layer

    

"""class KerasLayer(Layer):
"""
#Wrapper to keras layers
"""
def __init__(self):
super().__init__()
self.layers_module = tfkl

def set_config(self, config):
super().set_config(config)

class DienenLayer(Layer):
"""
#Wrapper to dienen custom layers
"""
def __init__(self):
super().__init__()
self.layers_module = dienen.layers

def set_config(self,config):
super().set_config(config)
"""