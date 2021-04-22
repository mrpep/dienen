from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from datetime import datetime
import os

from pyknife.aws import S3File

class SaveCheckpoints(Callback):
    def __init__(self, save_optimizer = True, frequency = 1, time_unit = 'epoch', save_best = True, monitor_metric = 'val_loss', monitor_criteria = 'auto', compression_level=3, save_initialization=False, save_last = True, clean_old=True, wandb_log=False):
        """
        Callback to save model parameters (aka checkpoints) regularly.
        
        args:
        save_optimizer:         bool (default=True), whether to include or not the optimizer state in the checkpoint.
                                It is recommended to keep training from a loaded checkpoint.
        frequency:              int (default=1). Interval (measured in time_unit) between 2 checkpoints are saved.
        time_unit:              str (default='epoch'). Can be 'epoch' or 'step', depending on if we want to save checkpoints
                                each frequency epochs or each frequency training steps.
        save_best:              bool (default=True). If True, epochs that do not have the best monitor_metric value are not saved.
        monitor_metric:         str (default='val_loss'). Indicates which metric is monitored when using save_best.
        monitor_criteria:       str (default='auto'). Can be 'max', if the monitor_metric is better with higher values, 'min' if it is
                                better with lower values, or 'auto', in which case it is decided by the program itself (this works for metrics
                                with 'acc' or 'fmeasure' in its name, in which case max criteria is used. When those 2 substrings don't appear 
                                in the metric, min criteria is used)
        compression_level:      int between 0 and 9 (default=3). 0 means no compression in the checkpoint files. Higher levels mean higher
                                compression ratios, with the cost of slower read/write times.
        save_initialization:    bool (default=False). Whether to save or not the initial weights (without training) as checkpoint. This can be
                                useful if we want to know how a randomly initialized model performs.
        save_last:              bool (default=True). Whether to always save the last checkpoint. This is useful if the training hangs and we
                                need to restart training.
        clean_old:              bool (default=True). If save_best and save_last are both True, then we want to erase all the checkpoints which
                                are not the best, but were saved because of using save_last. When clean_old is True, all the checkpoints but
                                the best and the last are erased at each callback call.
        wandb_log:              bool (default=False). If True, checkpoints are logged to Weights and Biases server as artifacts.
        """
        self.save_optimizer = save_optimizer
        self.frequency = frequency
        self.time_unit = time_unit
        self.save_best = save_best
        self.save_initialization = save_initialization
        self.monitor_metric = monitor_metric
        self.monitor_criteria = monitor_criteria
        self.compression_level = compression_level
        self.clean_old = clean_old
        self.save_last = save_last
        self.wandb_log = wandb_log

        if self.monitor_criteria == 'auto':
            if 'acc' in self.monitor_metric or self.monitor_metric.startswith('fmeasure'):
                self.monitor_criteria = 'max'
            else:
                self.monitor_criteria = 'min'
                
        if self.monitor_criteria == 'max':
            self.current_best = -np.Inf
        else:
            self.current_best = np.Inf
        
        self.step = 0
        self.epoch = 0
        self.current_metric = None
        
        self.checkpoints_history = []
    
    def on_epoch_end(self, batch, logs):
        if self.time_unit == 'epoch':
            self.current_metric = logs.get(self.monitor_metric, None)
            if isinstance(self.current_metric,tf.Tensor):
                self.current_metric = self.current_metric.numpy()
            epoch_saved = False
            if self.save_best:
                if self.current_metric:
                    if self.monitor_criteria == 'max' and self.current_metric>self.current_best:
                        self.current_best = self.current_metric
                        self.save(mode = 'epoch')
                        epoch_saved=True
                    elif self.monitor_criteria == 'min' and self.current_metric<self.current_best:
                        self.current_best = self.current_metric
                        self.save(mode = 'epoch')
                        epoch_saved=True
            if self.save_last and not epoch_saved:
                self.save(mode='epoch')                    
            if self.clean_old:
                if str(self.model_path).startswith('s3:'):
                    s3_ckpt_path = S3File(self.model_path,'checkpoints')
                    ckpt_path = Path(s3_ckpt_path.get_key())
                else:
                    s3_ckpt_path = None
                    ckpt_path = Path(self.model_path,'checkpoints')

                metadata = joblib.load(Path(ckpt_path,'metadata'))
                best_metric = self.current_best
                best_epoch_paths = []
                last_epoch_paths = []
                best_metadata_idx = -1
                last_metadata_idx = -1
                last_step = -1
                for i,w in enumerate(metadata):
                    if self.monitor_criteria == 'max' and w['metric_val']>=best_metric:
                        best_metric = w['metric_val']
                        best_epoch_paths = [w.get('weights_path'),w.get('opt_weights_path')]
                        best_metadata_idx = i
                    elif self.monitor_criteria == 'min' and w['metric_val']<=best_metric:
                        best_metric = w['metric_val']
                        best_epoch_paths = [w.get('weights_path'),w.get('opt_weights_path')]
                        best_metadata_idx = i
                    if w['step'] > last_step:
                        last_step = w['step']
                        last_epoch_paths = [w.get('weights_path'),w.get('opt_weights_path')]
                        last_metadata_idx = i
                paths_to_keep = best_epoch_paths + last_epoch_paths
                paths_to_keep = [k for k in paths_to_keep if k is not None]
                paths_to_keep += [str(Path(ckpt_path,'metadata').absolute())]
                all_files = [str(k.absolute()) for k in list(ckpt_path.rglob('*'))]
                to_delete = list(set([f for f in all_files if f not in paths_to_keep]))
                
                for f in to_delete:
                    if Path(f).exists():
                        os.remove(f)
                    if s3_ckpt_path is not None:
                        s3_f = S3File(str(s3_ckpt_path),Path(f).name)
                        if s3_f.exists():
                            s3_f.delete()

                metadata = [metadata[best_metadata_idx], metadata[last_metadata_idx]]
                if metadata[1]['weights_path'] == metadata[0]['weights_path']:
                    metadata.pop(0)

                joblib.dump(metadata,Path(ckpt_path,'metadata')) 
                if str(self.model_path).startswith('s3:'):
                    S3File(str(s3_ckpt_path),'metadata').upload(Path(ckpt_path,'metadata'))


        self.epoch += 1

    def on_train_begin(self,logs):
        print(logs)
        if self.save_initialization and self.epoch == 0:
            self.save(mode = 'epoch')
            self.epoch += 1
        
    def save(self, mode):
        
        if str(self.model_path).startswith('s3:'):
            s3_ckpt_path = S3File(self.model_path,'checkpoints')
            ckpt_path = Path(s3_ckpt_path.get_key())
            if not Path(ckpt_path,'metadata').exists() and S3File(str(s3_ckpt_path),'metadata').exists() and self.cache:
                S3File(s3_ckpt_path,'metadata').download(Path(ckpt_path,'metadata'))
        else:
            ckpt_path = Path(self.model_path,'checkpoints')

        if not ckpt_path.exists():
            ckpt_path.mkdir(parents=True,exist_ok=True)

        if Path(ckpt_path,'metadata').exists():
            self.checkpoints_history = joblib.load(Path(ckpt_path,'metadata'))
            
        weights = {}
        
        for layer in self.model.layers:
            if layer.name not in self.model.input_names:
                weights[layer.name] = [layer.get_weights(),[v.name for v in layer.variables]]
                
        if mode == 'step':
            current_step = self.step
        elif mode == 'epoch':
            current_step = self.epoch
        else:
            raise Exception("Unknown mode")
        
        checkpoint_history = {'mode': mode, 'step': current_step}
        checkpoint_history['time'] = datetime.now()
        
        if self.monitor_metric and self.current_metric:
            str_metric = "{}: {}".format(self.monitor_metric,self.current_metric)
            checkpoint_history['metric'] = self.monitor_metric
            checkpoint_history['metric_val'] = self.current_metric
        else:
            str_metric = ""
        filename = "{}-{}-{}".format(mode,current_step,str_metric)
        
        weights_dir = Path(ckpt_path,filename+'.weights')
        joblib.dump(weights, weights_dir, compress=self.compression_level)

        #Upload to S3
        if str(self.model_path).startswith('s3:'):
            S3File(str(s3_ckpt_path),filename+'.weights').upload(weights_dir)

        checkpoint_history['weights_path'] = str(weights_dir.absolute())
        if self.wandb_log:
            import wandb

            artifact = wandb.Artifact(filename.replace(': ','-') + '.weights', type='model')
            artifact.add_file(str(weights_dir))
            wandb.log_artifact(artifact)
            
        if self.save_optimizer:
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            if symbolic_weights:
                opt_weights = tf.keras.backend.batch_get_value(symbolic_weights)
            
                opt_dir = Path(ckpt_path,filename+'.opt')
                joblib.dump(opt_weights, opt_dir, compress=self.compression_level)
                checkpoint_history['opt_weights_path'] = str(opt_dir.absolute())

                #Upload to S3
                if str(self.model_path).startswith('s3:'):
                    S3File(str(s3_ckpt_path),filename+'.opt').upload(opt_dir)

                if self.wandb_log:
                    import wandb

                    artifact = wandb.Artifact(filename.replace(': ','-') + '.opt', type='model')
                    artifact.add_file(str(opt_dir))
                    wandb.log_artifact(artifact)

            else:
                checkpoint_history['opt_weights_path'] = None
            
        self.checkpoints_history.append(checkpoint_history)
        metadata_dir = Path(ckpt_path,'metadata')
        
        joblib.dump(self.checkpoints_history,metadata_dir, compress=self.compression_level)

        #Upload to S3
        if str(self.model_path).startswith('s3:'):
            S3File(str(s3_ckpt_path),'metadata').upload(metadata_dir)
        
    def on_batch_end(self, batch, logs):       
        if self.time_unit == 'step' and self.step%self.frequency == 0:
            self.current_metric = logs.get(self.monitor_metric, None)
            self.save(mode = 'step')
        self.step += 1