from tensorflow.keras.callbacks import Callback
from datetime import datetime
from pathlib import Path
import pandas as pd
import copy
import wandb
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

class WANDBLogger(Callback):
    """
    Log metrics to Weights and Biases server.

    args:

    loggers:        a dictionary containing different loggers configurations. The supported keys are
                    'Spectrograms' and 'TrainMetrics'. These log respectively, spectrograms of data in a given
                    model layer output, and the metrics calculated in the model.
                    The values consists of dictionaries with parameters of each logger.

                    Both loggers support 'freq' and 'unit' (which can be 'epoch' or 'step'), to indicate how often
                    data is logged.
                    Spectrograms supports:
                        in_layers:          indicates the layer/s that is the model input
                        out_layers:         indicates the name of the layer from which the spectrogram is calculated.
                        test_data:          data to use as input to obtain the spectrograms
                        plot_lims:          [vmin, vmax] of the plt.imshow function called (limits of the color)
                    TrainMetrics supports:
                        prefix:             (optional) string which is appended at the beginning of the metric name when logging.






    """
    def __init__(self, loggers=None, wandb_run = None):
        self.wandb_run = wandb_run
        self.step = 0
        self.epoch = 0
        self.loggers = loggers
        self.log_mapping = {'Spectrograms': self.log_spectrograms,
                            'TrainMetrics': self.log_train_metrics,
                            'ValidationMetrics': self.log_val_metrics}
        
    def log_spectrograms(self, params, logs):
        inputs = [self.model.get_layer(l).output for l in params.get('in_layers',None)]
        outs = [self.model.get_layer(l).output for l in params.get('out_layers',None)]
        

        predict_fn = tf.keras.backend.function(inputs=inputs,outputs=outs)
        plot_lims = params.get('plot_lims', [None, None])
        test_data = params.get('test_data',None)

        #if type(test_data).__name__ == 'BatchGenerator':
        #    x,y = test_data.__getitem__(0)

        x,y = test_data.__getitem__(0)

        out_names = params.get('out_layers',None)
        y_pred = predict_fn(x)

        for i in range(len(y_pred[0])): #i->instancia j-> activacion
            sample_plots = []
            for j in range(len(y_pred)):
                out_name = out_names[j]
                plt.figure()
                title = '{}'.format(out_name.replace('/','-'))
                plt.title(title)
                plt.imshow(np.squeeze(y_pred[j][i]).T,aspect='auto',origin='lower',vmin=plot_lims[0],vmax=plot_lims[1])
                sample_plots.append(wandb.Image(plt))
                plt.close()

            wandb.log({"sample_{}".format(i): sample_plots},step=self.step)

    def log_train_metrics(self, params, logs):
        prefix = params.get('prefix',None)
        logs_ = {}
        for k,v in logs.items():
            if not isinstance(v,float):
                v = v.numpy()
            if prefix is not None:
                logs_['{}_{}'.format(prefix,k)] = v
                logs_['{}_step'.format(prefix)] = self.step
            else:
                logs_['{}'.format(k)] = v

        logs_['learning_rate'] = self.model.optimizer._decayed_lr('float32').numpy()
        if prefix is not None:
            wandb.log(logs_)
        else:
            wandb.log(logs_,step=self.step)

    def log_val_metrics(self,params,logs):
        if 'custom_metrics' in params:
            from dienen.utils import get_modules, get_members_from_module
            import inspect

            metrics_module = params.get('metrics_module')
            if not isinstance(metrics_module,list):
                metrics_module = [metrics_module]

            metrics_module = get_modules(metrics_module)
            available_metrics = {}
            for metricmod in metrics_module:
                available_metrics.update(dict(get_members_from_module(metricmod,filters=[inspect.isclass,inspect.isfunction])))

            model_outs = [[self.model.predict(x), y] for x,y in params['validation_data']]
            y_pred = np.concatenate(np.array([x[0] for x in model_outs]))
            y_true = np.concatenate(np.array([x[1] for x in model_outs]))
            y_pred = y_pred[:len(params['validation_data'].idxs)]
            y_true = y_true[:len(params['validation_data'].idxs)]

            metric_results = {} 

            for metric in params['custom_metrics']:
                metric_type = metric.pop('type')
                metric_cls = available_metrics[metric_type](**metric)
                metric_results['val_{}'.format(metric_type)] = metric_cls.calculate(y_true,y_pred)

            wandb.log(metric_results,step=self.step)

        else:
            metrics = self.model.evaluate(params['validation_data'],return_dict=True)
            metrics = {'val_{}'.format(k): v for k,v in metrics.items()}
            wandb.log(metrics,step=self.step)
        
    def on_epoch_end(self, batch, logs):
        for log_type, log_params in self.loggers.items():
            if (log_params['unit'] == 'epoch') and (log_params['freq'] % self.step == 0):
                self.log_mapping[log_type](logs)

        if ('TrainMetrics' in self.loggers) and len(logs)>0:
            self.log_train_metrics(self.loggers['TrainMetrics'],logs)

        self.epoch += 1
        
    def on_batch_end(self, batch, logs):
        for log_type, log_params in self.loggers.items():
            if (log_params['unit'] == 'step') and (self.step % int(log_params['freq']) == 0):
                self.log_mapping[log_type](log_params, logs)

        self.step += 1
