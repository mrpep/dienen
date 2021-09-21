from tensorflow.keras.callbacks import Callback
from datetime import datetime
from pathlib import Path
import pandas as pd
import copy
import wandb
import tensorflow as tf
from scipy.special import logit, expit
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

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
                            'ValidationMetrics': self.log_val_metrics,
                            'Audios': self.log_audios}

    def on_train_begin(self, logs):
        self.model.metrics_log = {}

    def log_audios(self,params,logs):
        inputs = [self.model.get_layer(l).output for l in params.get('in_layers',None)]
        outs = [self.model.get_layer(l).output for l in params.get('out_layers',None)]
        
        predict_fn = tf.keras.backend.function(inputs=inputs,outputs=outs)

        test_data = params.get('test_data',None)
        log_y = params.get('log_gt',True)
        join_by = params.get('join_by', None)

        test_data.step = 0
        test_data.intraepoch_step = 0

        x,y = test_data.__getitem__(0)
        sr = params.get('sr',16000)

        y_pred = predict_fn(x)
        n_samples = y_pred[0].shape[0]
        batch_data = test_data.data.iloc[:n_samples].reset_index()

        if join_by is not None:
            for i,g in enumerate(batch_data[join_by].unique()):
                g_data = batch_data.loc[batch_data[join_by] == g]
                max_samples = g_data.end.max()
                g_audio = np.zeros((max_samples,))
                for logid, row in g_data.iterrows():
                    duration = int(row['end']) - int(row['start'])
                    window = np.concatenate([np.linspace(0,1,duration//2),np.linspace(1, 0,duration - duration//2)])
                    g_audio[int(row['start']):int(row['end'])] = g_audio[int(row['start']):int(row['end'])] + window*y_pred[0][int(logid)]
                if isinstance(sr, int):
                    sr_i = sr
                else:
                    sr_i = int(g_data.iloc[0][sr])

                wandb.log({g.split('/')[-1]: wandb.Audio(g_audio, caption=g.split('/')[-1], sample_rate=sr_i)})
  
    def log_spectrograms(self, params, logs):
        inputs = [self.model.get_layer(l).output for l in params.get('in_layers',None)]
        outs = [self.model.get_layer(l).output for l in params.get('out_layers',None)]
        
        predict_fn = tf.keras.backend.function(inputs=inputs,outputs=outs)
        plot_lims = params.get('plot_lims', [None, None])
        test_data = params.get('test_data',None)

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
                pred_i = np.squeeze(y_pred[j][i])
                if pred_i.ndim == 3:
                    pred_i = np.argmax(pred_i,axis=-1)
                plt.imshow(pred_i.T,aspect='auto',origin='lower',vmin=plot_lims[0],vmax=plot_lims[1])
                sample_plots.append(wandb.Image(plt))
                plt.close()

            wandb.log({"sample_{}".format(i): sample_plots},step=self.step)

    def log_train_metrics(self, params, logs):
        prefix = params.get('prefix',None)
        logs_ = {}
        for k,v in logs.items():
            if (not isinstance(v,float)) and (not isinstance(v,np.object)):
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
        return logs_

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

            params_metrics = copy.deepcopy(params['custom_metrics'])

            codebook_layers = list(set([m['codebook_layer'] for m in params_metrics if 'Codebook' in m['type']]))
            codebook_layers = {c: self.model.get_layer(c) for c in codebook_layers}
            codebook_layers_weights = {cl_name: {v.name: w for v, w in zip(cl.variables, cl.get_weights())} for cl_name, cl in codebook_layers.items()}

            model_outs = [[self.model.predict(x), y] for x,y in params['validation_data']]

            y_pred = np.array([x[0] for x in model_outs])
            y_true = np.array([x[1] for x in model_outs])
            if y_pred.ndim > 1:
                y_pred = np.concatenate(y_pred)
            if y_true.ndim > 1:
                y_true = np.concatenate(y_true)

            y_pred = y_pred[:len(params['validation_data']._index)]
            y_true = y_true[:len(params['validation_data']._index)]

            corrected_logit = None

            if 'train_priors' in params:
                if isinstance(params['train_priors'],str):
                    train_priors = joblib.load(Path(params['train_priors']).expanduser())
                elif isinstance(params['train_priors'],float):
                    train_priors = [params['train_priors']]*y_pred.shape[-1]
                corrected_logit = logit(y_pred) - logit(np.array(train_priors))[np.newaxis,:]
            if 'validation_priors' in params:
                if isinstance(params['validation_priors'],str):
                    validation_priors = joblib.load(Path(params['validation_priors']).expanduser())
                elif isinstance(params['validation_priors'],float):
                    validation_priors = [params['validation_priors']]*y_pred.shape[-1]
                corrected_logit = corrected_logit + logit(np.array(validation_priors))[np.newaxis,:]

            if corrected_logit is not None:
                y_pred = expit(corrected_logit)

            metric_results = {} 

            for metric in params_metrics:
                metric_type = metric.pop('type')
                if inspect.isclass(available_metrics[metric_type]):
                    metric_cls = available_metrics[metric_type](**metric)
                    if 'Codebook' in metric_type:
                        metric_cls.codebook_layers = codebook_layers
                        metric_cls.validation_data = params['validation_data']
                        metric_cls.model = self.model
                    mres = metric_cls(y_true,y_pred)
                elif inspect.isfunction(available_metrics[metric_type]):
                    mres = available_metrics[metric_type](y_true,y_pred)
                else:
                    raise Exception('Unrecognized metric')

                for k,v in mres.items():
                    if v.ndim == 0:
                        metric_results['val_{}'.format(k)] = v
                    elif v.ndim == 1:
                        labels = None
                        if 'labels' in params:
                            if isinstance(params['labels'],str):
                                labels = joblib.load(Path(params['labels']).expanduser())
                            else:
                                labels = params['labels']
                        if len(v)>100:
                            for i in range(len(v)//100):
                                plt.figure(figsize=(20,10))
                                sns.barplot(x=labels[i*100:(i+1)*100], y=v[i*100:(i+1)*100])
                                plt.xticks(rotation=90)
                                wandb.log({'val_{}_{}'.format(k,i): wandb.Image(plt)},step=self.step)
                            if len(v)%100 > 0:
                                plt.figure(figsize=(20,10))
                                sns.barplot(x=labels[100*(len(v)//100):], y=v[100*(len(v)//100):])
                                plt.xticks(rotation=90)
                                wandb.log({'val_{}_{}'.format(k,len(v)//100): wandb.Image(plt)},step=self.step)
                        else:
                            plt.figure(figsize=(20,10))
                            if len(labels) == len(v):
                                sns.barplot(x=labels, y=v)
                            else:
                                sns.barplot(x=np.arange(len(v)),y=v)
                            plt.xticks(rotation=90)
                            wandb.log({'val_{}'.format(k): wandb.Image(plt)},step=self.step)
                    elif v.ndim == 2:
                        from IPython import embed
                        embed()
                    elif v.ndim == 3:
                        for i, v_i in enumerate(v):
                            plt.figure(figsize=(10,10))
                            plt.imshow(v_i,aspect='auto',origin='lower')
                            wandb.log({'val_{}_{}'.format(k,i): wandb.Image(plt)},step=self.step)
            wandb.log(metric_results,step=self.step)            
        else:
            metric_results = self.model.evaluate(params['validation_data'],return_dict=True)
            metric_results = {'val_{}'.format(k): v for k,v in metric_results.items()}
            wandb.log(metric_results,step=self.step)
        return metric_results
        
    def on_epoch_end(self, batch, logs):
        if logs is not None:
            logged_metrics = None
            for log_type, log_params in self.loggers.items():
                if (log_params['unit'] == 'epoch') and ((self.epoch + 1) % int(log_params['freq']) == 0):
                    logged_metrics = self.log_mapping[log_type](log_params,logs)

            if ('TrainMetrics' in self.loggers) and len(logs)>0:
                logged_metrics = self.log_train_metrics(self.loggers['TrainMetrics'],logs)
            if logged_metrics is not None:
                logs.update(logged_metrics)

            self.model.metrics_log.update(logs)

        self.epoch += 1
        
    def on_batch_end(self, batch, logs):
        for log_type, log_params in self.loggers.items():
            if (log_params['unit'] == 'step') and ((self.step + 1) % int(log_params['freq']) == 0):
                logged_metrics = self.log_mapping[log_type](log_params, logs)
                if logged_metrics is not None:
                    logs.update(logged_metrics)
        self.model.metrics_log.update(logs)

        self.step += 1
