from .architecture import Architecture
from .training import TrainingNode

from dienen.config_processors import ArchitectureConfig
from dienen.config_processors.utils import set_config_parameters
from dienen.utils import get_config, get_available_gpus
from dienen.core.file import load_weights

from ruamel_yaml import YAML

import copy
from datetime import datetime
import joblib
from pathlib import Path
import tensorflow as tf
from kahnfigh import Config, shallow_to_deep, shallow_to_original_keys

class Model():
    def __init__(self,config,logger=None):
        """
        Main class of the dienen library. It represents the model, which is built from a configuration file.
        config: can be a string or pathlib.Path pointing to a .yaml file, a dictionary or a kahnfigh Config.
        logger: optionally, a logger can be supplied to log all information related to dienen model.
        """

        config = Config(config,safe=False)

        self.original_config = config
        self.config = copy.deepcopy(config)
        self.core_model = None
        self.architecture_config = None
        self.model_path = None
        self.name = self.config['Model'].get('name',datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        if not self.model_path:
            self.model_path = self.config['Model'].get('path','{}'.format(self.name))
        self.weights = None
        self.optimizer_weights = None
        self.extra_data = None
        self.modules = self.config.get('Module',[])
        self.gpu_config = self.config.get('gpu_config',{'device': 'auto', 'allow_growth': True})
        self.cache = True
        self.logger = logger
        self.input_shapes = None
        self.output_shapes = None

        gpu_device = self.gpu_config.get('device','auto')
        if gpu_device == 'auto':
            gpu, mem = get_available_gpus()
            gpu_device = int(gpu)
            if self.logger:
                self.logger.info("Automatically selected device {} with {} available memory".format(gpu_device,mem))
        gpu_growth = self.gpu_config.get('allow_growth',True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                if len(gpus) < gpu_device:
                    raise Exception('There are only {} available GPUs and the {} was requested'.format(len(gpus),gpu_device))
                tf.config.experimental.set_visible_devices(gpus[gpu_device], 'GPU')
            except RuntimeError as e:
                raise Exception('Failed setting GPUs. {}'.format(e))
        if gpu_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, gpu_growth)
                except RuntimeError as e:
                    raise Exception('Failed setting GPU dynamic memory allocation. {}'.format(e))

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        if self.logger:
            self.logger.debug("Physical GPUs: {}. Logical GPUs: {}".format(len(gpus), len(logical_gpus)))

        self.externals = self.config['Model'].get('External', None)

    def build(self, return_tensors=False, processed_config=None,input_names=None,output_names=None):
        """
        Builds the specified model, returning a tensorflow model.

        optional_args:
        return_tensors:     if True, returns (architecture, keras_model), with architecture being a Architecture object,
                            which can be used to access each layer output or manually build a new model from it.
        processed_config:   optionally pass an external config, which will replace the internal model configuration.
        input_names:        a list of strings, specifying the names of the input layers. These will replace the inputs 
                            specified in the configuration file.
        output_names:       a list of strings, specifying the names of the output layers. These will replace the outputs 
                            specified in the configuration file.
        """

        architecture_config = self.config['Model']['Architecture']
        self.architecture_config = ArchitectureConfig(architecture_config, self.externals, logger=self.logger)

        if input_names is None:
            input_names = self.config['Model']['inputs']
        if output_names is None:
            output_names = self.config['Model']['outputs']

        #Automatic shape assignment to inputs and outputs (in this case only Dense outputs are considered)
        if self.input_shapes is not None:
            for i,in_name in enumerate(input_names):
                for layer in self.architecture_config.config:
                    class_name = list(layer.keys())[0]
                    if class_name == 'Input' and layer[class_name].get('name',None) == in_name and 'shape' not in layer[class_name]:
                        layer[class_name]['shape'] = self.input_shapes[i]
                        if self.logger:
                            self.logger.info('Layer {}: automatically set input shape as: {}'.format(in_name,self.input_shapes[i]))
            
            for i,out_name in enumerate(output_names):
                for layer in self.architecture_config.config:
                    class_name = list(layer.keys())[0]
                    if class_name == 'Dense' and layer[class_name].get('name',None) == out_name and 'units' not in layer[class_name]:
                        layer[class_name]['units'] = self.output_shapes[i][0]                
                        if self.logger:
                            self.logger.info('Layer {}: automatically set units as: {}'.format(out_name,self.output_shapes[i][0]))
        
        nn = Architecture(self.architecture_config,inputs=input_names,outputs=output_names,externals=self.externals,processed_config=processed_config)
        self.core_model = nn

        if return_tensors:
            return nn.model, nn
        else:
            return nn.model

    def fit(self, data = None, from_epoch = -1, validation_data = None):
        """
        Trains the model.
        data:               can be any object which is accepted by tf.keras.Model.fit()
        from_epoch:         specifies initial epoch
        validation_data:    can be any object which is accepted by the validation_data kwarg of tf.keras.Model.fit()
        """
        self.training_node = TrainingNode(self.config['Model']['Training'], modules = self.modules, logger=self.logger)
        if data is None:
            data = self.train_data
        if validation_data is None:
            validation_data = self.validation_data
        if self.weights:
            self.training_node.set_weights(self.weights)
        if self.optimizer_weights:
            self.training_node.set_optimizer_weights(self.optimizer_weights)
        if self.extra_data:
            self.training_node.set_extra_data(self.extra_data)
        if not self.core_model:
            self.build()

        if self.logger:
            self.logger.info('Model Summary: \n{}'.format(self.core_model.model.summary()))
        else:
            print(self.core_model.model.summary())
       
        self.training_node.fit(self.core_model.model, data, self.model_path, validation_data = validation_data, from_epoch = from_epoch, cache=self.cache)


    def get_optimizer(self):
        """
        Returns the optimizer state
        """
        opt_weights = None
        if hasattr(self.core_model.model.optimizer,'weights'):
            symbolic_weights = getattr(self.core_model.model.optimizer, 'weights')
            if symbolic_weights:
                opt_weights = tf.keras.backend.batch_get_value(symbolic_weights)

        return opt_weights

    def get_weights(self):
        """
        Get the weights of the model.
        A dictionary is returned with the layer names as keys and the weights as values.
        """
        if self.core_model is not None:
            weights = {}
            for layer_name in self.core_model.processed_config:
                weights[layer_name] = self.core_model.model.get_layer(layer_name).get_weights()
            return weights
        else:
            return None

    def modify(self,mods,inputs=None,outputs=None):
        """
        Allows to make modifications to the model. It will create a modified configuration file, build the corresponding model and
        set the weights of the current model when possible.

        mods:    a list of dictionaries. Each dictionary can have as key: delete (in this case the value, which is a path, 
                 is deleted from config), or a config path (in this case, the path value is replaced by the value of the dictionary).
        inputs:  a list with the names of the inputs
        outputs: a list with the names of the outputs
        """
        model_weights = self.get_weights()
        yaml_loader = YAML()
        m_conf = Config(self.core_model.processed_config)
        original_keys = list(m_conf.keys())
        deep_conf = Config(shallow_to_deep(m_conf))
        for mod in mods:
            mod_key = list(mod.keys())[0]
            mod_value = mod[mod_key]
            if mod_key == 'delete':
                deep_conf.pop(mod_value)
                if mod_value in original_keys:
                    original_keys.remove(mod_value)
            elif '*' in mod_key:
                mod_key = mod_key.lstrip('/')
                found_paths = [k for k in deep_conf.to_shallow().keys() if fnmatch.fnmatch(k,mod_key)]
                for k in found_paths:
                    k = k.replace('.','/')
                    if isinstance(mod_value,str):
                        deep_conf[k] = yaml_loader.load(mod_value)
                    else:
                        deep_conf[k] = mod_value
            else:
                mod_key = mod_key.replace('.','/')
                if mod_key.split('/')[0] not in deep_conf.keys(): #This means we are adding a new layer
                    layer_name = mod_key.split('/')[0]
                    original_keys.append(layer_name)
                    deep_conf['{}/name'.format(layer_name)]=layer_name
                if isinstance(mod_value,str):
                    deep_conf[mod_key] = yaml_loader.load(mod_value)
                else:
                    deep_conf[mod_key] = mod_value
        new_model_architecture = shallow_to_original_keys(deep_conf.to_shallow(),original_keys)
        model = self.build(processed_config=new_model_architecture,input_names=inputs,output_names=outputs)
        layer_names = [l.name for l in model.layers]
        for k,v in model_weights.items():
            if k in layer_names:
                layer = model.get_layer(k)
                layer.set_weights(v)
        self.core_model.model = model

    def predict(self,data,output = 'output'):
        """
        Makes a prediction over the given data. It can return not only the model outputs but also the intermediate
        activations.

        data:       accepts the same objects than tf.keras.Model.predict()
        output:     (default: 'output'). If 'output', only the model output is returned. If 'all', activations from
                    all model layers are returned. A list of strings can be given to indicate which layers activations
                    are returned.
        """
        if output == 'output':
            return self.core_model.model.predict(data)
        else:
            if output == 'all':
                output = [layer.name for layer in self.core_model.model.layers]
            else: 
                if not isinstance(output,list):
                    output = [output]
            
            outputs = []
            output_names = []
            inputs = []
            input_names = []
            for layer in self.core_model.model.layers:
                if hasattr(layer,'is_placeholder'):
                    inputs.append(layer.output)
                    input_names.append(layer.name)
                elif layer.name in output:
                    outputs.append(layer.output)
                    output_names.append(layer.name)
                else:
                    pass
            predict_fn = tf.keras.backend.function(inputs = inputs,outputs=outputs)
            activations = predict_fn(data)
            activations = {name: act for name, act in zip(output_names,activations)}
            return activations

    def save_model(self,filename,save_optimizer=False,extra_data=None):
        """
        Save the model
        filename:       path where to save the model
        save_optimizer: if True, the optimizer state is saved. This is necessary to keep training a model
        extra_data:     metadata to save along with the model (its keys are added to the keys of the saved dictionary)
        """
        model_output = self._serialize_model(save_optimizer=save_optimizer,extra_data=extra_data)
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        joblib.dump(model_output,str(filename.absolute()))

    def save_weights(self, filename):
        """
        Save the model weights in a file.

        filename:       path where to save the model weights.
        """
        weights = self.get_weights()
        if weights:
            filename = Path(filename)
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)
            joblib.dump(weights,str(filename.absolute()))
        else:
            raise Exception('Model does not have weights')

    def _serialize_model(self, save_optimizer=False, extra_data=None):
        model_output = {}
        model_output['weights'] = self.get_weights()
        original_config = Config(self.original_config)
        for p in original_config.all_paths():
            if type(original_config[p]).__name__ == 'BatchGenerator':
                original_config[p] = original_config[p].data_processor_config
        model_output['original_config'] = original_config
        if self.architecture_config:
            model_output['hierarchy'] = self.architecture_config.hierarchy
        if self.core_model:
            model_output['unfolded_config'] = self.core_model.processed_config
            if save_optimizer:
                model_output['optimizer_state'] = self.get_optimizer()
            if extra_data:
                model_output.update(extra_data)
        model_output['input_shapes'] = self.input_shapes
        model_output['output_shapes'] = self.output_shapes

        return model_output

    def set_data(self,train,validation=None):
        """
        Set the training and validation dataset. Doing this before calling build, allows to do automatic inference of
        the model inputs and outputs shapes.

        train:      a generator for model training
        validation: a generator for model validation
        """
        self.train_data = train
        self.validation_data = validation

        if isinstance(train,tuple):
            x = train[0][:16]
            y = train[1][:16]
        else:
            x, y = train.__getitem__(0)

        #For automatic shape inference
        if isinstance(x,list):
            self.input_shapes = [x_i.shape[1:] for x_i in x]
        else:
            self.input_shapes = [x.shape[1:]]

        if isinstance(y,list):
            self.output_shapes = [x_i.shape[1:] for x_i in y]
        else:
            self.output_shapes = [y.shape[1:]]

    def set_extra_data(self,data):
        """
        Supply extra metadata to the model
        """
        self.extra_data = data

    def set_logger(self, logger):
        """
        Sets a logger for the dienen model
        """
        self.logger = logger

    def set_model_path(self,path):
        """
        Sets the global model path, where checkpoints will be saved.
        """
        self.model_path = path

    def set_optimizer_weights(self,weights):
        """
        Set the optimizer weights to the given values

        weights:        can be a string or a pathlib Path with the path to a weights file, or a ?
        """
        if isinstance(weights,str) or isinstance(weights,Path):
            weights = joblib.load(weights)
        self.optimizer_weights = weights
        if hasattr(self,'core_model') and self.core_model.model.optimizer is not None:
            self.core_model.model.optimizer._create_all_weights(self.core_model.model.trainable_variables)
            self.core_model.model.optimizer.set_weights(weights)

    #def set_parameters(self, parameters):
    #    self.config = set_config_parameters(self.original_config,parameters)

    def set_weights(self,weights):
        """
        Set the model weights.

        weights:       can be a string or a pathlib Path with the path to a weights file, or a dictionary of weights.
        """
        if isinstance(weights,str) or isinstance(weights,Path):
            weights = joblib.load(weights)
        self.weights = weights
        if hasattr(self,'core_model'):
            for k,v in weights.items():
                layer = self.core_model.model.get_layer(k)
                layer.set_weights(v[0])

    def load_weights(self,strategy='min'):
        """
        Reads the checkpoints metadata saved, and automatically selects the best weights and sets them for the model.

        strategy:       (default='min'). If 'min', the checkpoint with minimum value of the monitored metric is used.
                        If 'max', the checkpoint with maximum monitored metric is used instead.
        """


        checkpoints_metadata_path = Path(self.model_path,'checkpoints','metadata')
        if checkpoints_metadata_path.exists():
            checkpoints_metadata = joblib.load(checkpoints_metadata_path)
            metric_vals = [c['metric_val'] for c in checkpoints_metadata]
            if strategy == 'min':
                idx = metric_vals.index(min(metric_vals))
            elif strategy == 'max':
                idx = metric_vals.index(max(metric_vals))

            weights_path = checkpoints_metadata[idx]['weights_path']
            opt_path = checkpoints_metadata[idx]['opt_weights_path']
            self.set_weights(weights_path)
            self.set_optimizer_weights(opt_path)

    def __getstate__(self):
        return self._serialize_model(save_optimizer=True, extra_data=self.extra_data)

    def __setstate__(self,model_data):
        self.__init__(model_data['original_config'])
        model_data.pop('original_config')
        self.logger = None
        self.input_shapes = model_data['input_shapes']
        self.output_shapes = model_data['output_shapes']
        self.build()
        if 'weights' in model_data:
            load_weights(model_data['weights'],self.core_model.model,has_variable_names=False)
            model_data.pop('weights')
        if 'optimizer_state' in model_data:
            self.set_optimizer_weights(model_data['optimizer_state'])
            model_data.pop('optimizer_state')
        self.set_extra_data(model_data)

    def clear_session(self):
        tf.keras.backend.clear_session()

    def set_seed(self, seed=1234):
        tf.random.set_seed(seed)

        


        

        

