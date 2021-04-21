import inspect
import sys
import importlib.util
from pathlib import Path
import tensorflow
import nvgpu
import numpy as np
import pickle
from ruamel_yaml import YAML
from .modiuls import get_modules, get_members_from_module
from kahnfigh import Config

def get_class_signature(elem, ignore_keys = None):
    """
    Returns the args and kwargs of a class constructor.
    Arguments:
        elem: a class to get info from
        ignore_keys: args and kwargs to ignore. By default: ['self','args','kwargs'].
    Outputs:
        (arg_list, kwarg_list): arguments and keyword arguments accepted by the class constructor.
    """
    if ignore_keys is None:
        ignore_keys = ['self','args','kwargs']

    if inspect.isclass(elem):
        parent_classes = list(elem.__bases__)
        elems = [elem] + parent_classes
    else:
        elems = [elem]

    arg_list = []
    kwarg_list = []
        
    for i,elem in enumerate(elems):
        if '__init__' in elem.__dict__:
            signature = inspect.signature(elem.__dict__['__init__'])
        else:
            signature = inspect.signature(elem)

        keys = list(signature.parameters.keys())
        keys_filtered = [key for key in keys if key not in ignore_keys]

        arg_list_i = []
        kwarg_list_i = []

        for key in keys_filtered:
            if signature.parameters[key].default == inspect._empty:
                if i == 0:
                    arg_list_i.append(key)
            else:
                kwarg_list_i.append(key)

        if len(arg_list_i) == 0 and len(kwarg_list_i) == 0 and i == 0:
            kwarg_list_i.extend(inspect.getfullargspec(elem.__init__).args)
        arg_list.extend([i for i in arg_list_i if i not in arg_list])
        kwarg_list.extend(kwarg_list_i)

    return arg_list, list(set(kwarg_list))

def get_classes_in_module(module):
    """
    Returns a dictionary containing all the available classes in a module.
    Arguments:
        module: a module from which to extract available classes
    Outputs:
        returns a dictionary containing class names as keys and class objects as values. 
    """

    clsmembers = inspect.getmembers(module, inspect.isclass)
    clsmembers_dict = {cls[0]:cls[1] for cls in clsmembers}

    keras_names = ['tensorflow_core.python.keras.api._v2.keras.layers',
                   'tensorflow_core.keras.layers',
                   'tensorflow.keras.layers']

    if module.__name__ in keras_names:
        clsmembers_dict['Input'] = tensorflow.keras.layers.Input

    return clsmembers_dict

def import_file(module_path):
    """
    Arguments:
        module_path: path to a .py file
    Returns:
        the module in module_path
    """

    module_path = Path(module_path)
    sys.path.append(str((module_path.absolute()).parent))
    module = importlib.import_module(module_path.stem)

    return module

def get_available_gpus():
    gpu_info = nvgpu.gpu_info()
    gpu_device = "-1"
    gpu_mem = 0
    for device in gpu_info:
        available_mem = device['mem_total'] - device['mem_used']
        if available_mem>gpu_mem:
            gpu_mem = available_mem
            gpu_device = device['index']

    return gpu_device, gpu_mem

def layer_with_parameter_in_config(config,parameter_key,parameter_value):
    for i, layer in enumerate(config):
        layer_dict = layer[list(layer.keys())[0]]
        layer_param = layer_dict.get(parameter_key,None)
        if layer_param == parameter_value:
            return i,layer
    
    return None

def numeric_to_one_hot(x):
    y = np.zeros((x.size, x.max()+1))
    y[np.arange(x.size),x] = 1
    
    return y

def name_layers(layer_names,architecture,prefix=''):
    to_list = False
    if not isinstance(architecture,list):
        architecture = [architecture]
        to_list = True
        
    for i,layer in enumerate(architecture):
        layer_type = list(layer.keys())[0]
        layer_config = layer[layer_type]
        if 'name' not in layer_config.keys():
            k = 0
            automatic_name = prefix + layer_type + '_' + str(k)
            while (automatic_name in layer_names) or (automatic_name in layer_names):
                k = k + 1
                automatic_name = prefix + layer_type + '_' + str(k)
            layer_names.append(automatic_name)
            layer_config['name'] = automatic_name
            architecture[i][layer_type] = layer_config
        else:
            layer_names.append(layer_config['name'])
    
    if to_list:
        architecture = architecture[0]
            
    return architecture,layer_names

def set_missing_inputs(architecture_config, last_layer = None):
    #If an input is not supplied for a layer, automatically use previous layer as input

    for i,layer in enumerate(architecture_config):
        layer_type = list(layer.keys())[0]
        layer_config = layer[layer_type]
        if 'input' not in layer_config.keys():
            layer_config['input'] = last_layer
            architecture_config[i][layer_type] = layer_config
        last_layer = layer_config['name']
        
    return architecture_config

def get_config(filename):
    config = Config(filename,safe=False)
    return config
