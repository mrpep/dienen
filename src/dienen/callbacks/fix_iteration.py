import tensorflow as tf

class UseOptimizerIterationAsTrainStep(tf.keras.callbacks.Callback):
    #Suggested in https://stackoverflow.com/questions/65031435/is-it-possible-to-restore-global-step-in-keras
    #by Tim Brychcy
    def on_train_begin(self, logs=None):
        self.model._train_counter.assign(self.model.optimizer.iterations)