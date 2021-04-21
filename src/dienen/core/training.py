from dienen.utils import get_modules, get_members_from_module
import inspect
from dienen.core.node import DienenNode
from pathlib import Path
import joblib
from .file import load_weights
import tensorflow as tf

class TrainingNode(DienenNode):
    def __init__(self,config, modules=None, logger=None):
        self.valid_nodes = ['loss','optimizer','schedule','n_epochs','workers','custom_fit','metrics']
        self.required_nodes = ['loss','optimizer']
        super().set_config(config)
        self.optimizer_weights = None
        self.weights = None
        self.modules = modules
        self.extra_data = None
        self.logger = logger

    def get_loss_fn(self):
        loss_params = self.config.get('loss', None)
        loss_module_names = ['tensorflow.keras.losses']
        #Agregar para levantar modulos externos
        loss_module_names = loss_module_names + self.modules
        loss_modules = get_modules(loss_module_names)
        
        available_losses = {}
        for lossmod in loss_modules:
            available_losses.update(dict(get_members_from_module(lossmod,filters=[inspect.isclass,inspect.isfunction])))

        if isinstance(loss_params,str):
            loss_fn = available_losses.get(loss_params)
        elif isinstance(loss_params,dict):
            loss_fn = available_losses[loss_params['type']]
            loss_params.pop('type')
            loss_fn = loss_fn(**loss_params)

        return loss_fn

    def get_metrics(self):
        metric_params = self.config.get('metrics',None)
        metric_module_names = ['tensorflow.keras.metrics']
        metric_module_names = metric_module_names + self.modules
        metric_modules = get_modules(metric_module_names)

        available_metrics = {}
        for metricmod in metric_modules:
            available_metrics.update(dict(get_members_from_module(metricmod,filters=[inspect.isclass,inspect.isfunction])))
        
        if metric_params:
            if not isinstance(metric_params,list):
                metric_params = [metric_params]
            metric_objs = []
            for m in metric_params:
                if isinstance(m, str):
                    if m in available_metrics:
                        metric_objs.append(available_metrics.get(m))
                    else:
                        raise Exception('{} is not an available metric. Available metrics are {}'.format(m,list(available_metrics.keys())))
                elif isinstance(m, dict):
                    if m['type'] in available_metrics:
                        metric_cls = available_metrics[m['type']]
                        m.pop('type')
                        metric_objs.append(metric_cls(**m))
                    else:
                        raise Exception('{} is not an available metric. Available metrics are {}'.format(m,list(available_metrics.keys())))

            return metric_objs

        else:
            return None

    def set_extra_data(self,data = None):
        self.extra_data = data

    def set_optimizer_weights(self,weights):
        if isinstance(weights,str) or isinstance(weights,Path):
            self.optimizer_weights = joblib.load(weights)
        else:
            self.optimizer_weights = weights

    def set_weights(self,weights):
        if isinstance(weights,str) or isinstance(weights,Path):
            self.weights = joblib.load(weights)
        else:
            self.weights = weights

    def get_optimizer(self):
        optimizer_params = self.config.get('optimizer',None)
        optimizer_modules_names = ['tensorflow.keras.optimizers']
        optimizer_modules_names = optimizer_modules_names + self.modules
        optimizer_modules = get_modules(optimizer_modules_names)
        
        available_optimizers = {}
        for optmod in optimizer_modules:
            available_optimizers.update(get_members_from_module(optmod,filters=[inspect.isclass]))

        if isinstance(optimizer_params,str):
            optimizer = available_optimizers.get(optimizer_params)
        elif isinstance(optimizer_params,dict):
            optimizer = available_optimizers[optimizer_params['type']]
            optimizer_params.pop('type')
            if 'dynamic_loss_scaling' in optimizer_params:
                dynamic_loss_scaling = optimizer_params.pop('dynamic_loss_scaling')
            else:
                dynamic_loss_scaling = True
            #Check if lr is a schedule:
            if optimizer_params.get('learning_rate',None) in available_optimizers.keys():
                lr_schedule = available_optimizers[optimizer_params['learning_rate']]
                if optimizer_params.get('learning_rate_args',None):
                    optimizer_params['learning_rate'] = lr_schedule(**optimizer_params['learning_rate_args'])
                    optimizer_params.pop('learning_rate_args')
                else:
                    optimizer_params['learning_rate'] = lr_schedule()

            optimizer = optimizer(**optimizer_params)
            #if dynamic_loss_scaling: #No anda con tf keras 2.4
                #self.logger.info('Dynamic loss scaling enabled')
            #    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        return optimizer

    def get_callbacks(self):
        schedule_params = self.config.get('schedule',None)
        cb_modules_names = ['tensorflow.keras.callbacks','dienen.callbacks']
        cb_modules_names = cb_modules_names + self.modules
        cb_modules = get_modules(cb_modules_names)
        #Agregar para levantar modulos externos
        available_cb = {}
        for cbmod in cb_modules:
            available_cb.update(get_members_from_module(cbmod,filters=[inspect.isclass]))

        #cb_list = [available_cb[k](**v) for k,v in schedule_params.items()]
        #cb_list = [cbk[list(cbk.keys())[0]] for cbk in schedule_params]

        if isinstance(schedule_params,list):
            cb_keys = [list(cbk.keys())[0] for cbk in schedule_params]
            cb_list = [available_cb[k](**v[k]) for k,v in zip(cb_keys,schedule_params)]
        elif isinstance(schedule_params,dict):
            cb_list = [available_cb[k](**v) for k,v in schedule_params.items()]
        elif schedule_params is None:
            cb_list = []

        for cb in cb_list:
            cb.model_path = self.model_path
            cb.extra_data = self.extra_data
            cb.cache = self.cache

        return cb_list

    def fit(self, keras_model, data, output_path, from_epoch=0, validation_data=None, cache=True):
        self.model_path = output_path
        self.cache = cache
        n_epochs = self.config.get('n_epochs',10)
        calculate_initial_loss = self.config.get('calculate_initial_loss',False)
        loss_fn = self.get_loss_fn()
        optimizer = self.get_optimizer()
        cb_list = self.get_callbacks()
        metrics = self.get_metrics()
        for cb in cb_list: 
            cb.epoch = from_epoch
            cb.data = data

        keras_model.compile(optimizer=optimizer,loss=loss_fn,metrics=metrics)

        if self.weights:
            load_weights(self.weights,keras_model)
        if self.optimizer_weights:
            keras_model.optimizer._create_all_weights(keras_model.trainable_variables)
            keras_model.optimizer.set_weights(self.optimizer_weights)

        if calculate_initial_loss and validation_data:
            if self.logger:
                self.logger.info('Initial Loss: {}'.format(keras_model.evaluate(validation_data)))

        n_workers = self.config.get('workers',1)
        if n_workers > 1:
            use_multiprocessing = True
        else:
            use_multiprocessing = False
        
        custom_fit = self.config.get('custom_fit', None)
        if hasattr(data,'epoch'):
            data.epoch = from_epoch
        if hasattr(data,'on_train_begin'):
            data.on_train_begin()
        if custom_fit:
            modules = get_modules(self.modules)
            custom_fit_fn = None
            for module in modules:
                available_fns = get_members_from_module(module,filters=[inspect.isfunction])
                if custom_fit in available_fns:
                    custom_fit_fn = available_fns[custom_fit]
                    break
            if custom_fit_fn:
                custom_fit_fn(data,keras_model,callbacks=cb_list,initial_epoch=from_epoch,epochs=n_epochs)
            else:
                raise Exception('{} not found'.format(custom_fit))

        else:
            if isinstance(data,list) or isinstance(data,tuple):
                if validation_data is not None:
                    validation_data = tuple(validation_data)
                keras_model.fit(x=data[0],y=data[1],initial_epoch=from_epoch,epochs=n_epochs,
                    callbacks=cb_list, validation_data=validation_data, use_multiprocessing = use_multiprocessing,
                    workers = n_workers, shuffle=False)
            else:
                keras_model.fit(data,initial_epoch=from_epoch,epochs=n_epochs,
                    callbacks=cb_list, validation_data=validation_data, use_multiprocessing = use_multiprocessing,
                    workers = n_workers, shuffle=False)

