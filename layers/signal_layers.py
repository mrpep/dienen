import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import copy

class Frame(tfkl.Layer):
    def __init__(self,win_size=None,hop_size=None,pad_end=False,pad_value=0,axis=-1,name=None, trainable=False):
        super(Frame, self).__init__(name=name)
        self.frame_args = {'ws':win_size,
                           'hs':hop_size,
                           'pad':pad_end,
                           'padv':pad_value,
                           'axis':axis}

    def call(self,x):
        return tf.signal.frame(x,
            frame_length=self.frame_args['ws'],
            frame_step=self.frame_args['hs'],
            pad_end=self.frame_args['pad'],
            pad_value=self.frame_args['padv'],
            axis=self.frame_args['axis'])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'win_size': self.frame_args['ws'],
            'hop_size': self.frame_args['hs'],
            'pad_end': self.frame_args['pad'],
            'pad_value': self.frame_args['padv'],
            'axis': self.frame_args['axis']
        })
        return config

class GetPatches(tfkl.Layer):
    def __init__(self,patch_size=None,name=None, trainable=False):
        super(GetPatches, self).__init__(name=name)
        self.patch_size = patch_size
        
    def call(self,x):
        #x (w,h,ch)
        ch = x.shape[3]
        h = x.shape[2]
        w = x.shape[1]
        h_i = self.patch_size[0]
        rows = h//h_i
        w_i = self.patch_size[1]
        cols = w//w_i
        
        permute_1 = tfkl.Permute((3,1,2))(x) #(ch,w,h)
        reshape_1 = tfkl.Reshape((ch,w,rows,h_i))(permute_1) #(ch,w,rows,h_i)
        permute_2 = tfkl.Permute((1,3,4,2))(reshape_1) #(ch,rows,h_i,w)
        reshape_2 = tfkl.Reshape((ch,rows,h_i,cols,w_i))(permute_2) #(ch,rows,h_i,cols,w_i)
        final_permute = tfkl.Permute((2,4,3,5,1))(reshape_2)
        final_reshape = tfkl.Reshape((rows*cols,h_i,w_i,ch))(final_permute)

        return final_reshape

class ImageFromPatches(tfkl.Layer):
    def __init__(self,rows=None,cols=None,name=None, trainable=False):
        super(ImageFromPatches, self).__init__(name=name)
        self.rows = rows
        self.cols = cols
        
    def call(self,x):
        #x (w,h,ch)
        h_i = x.shape[2]
        w_i = x.shape[3]
        ch = x.shape[4]

        x = tfkl.Reshape((self.rows,self.cols,h_i,w_i,ch))(x)
        x = tfkl.Permute((2,4,5,1,3))(x) #c,w_i,ch,r,h_i
        x = tfkl.Reshape((self.cols,w_i,ch,self.rows*h_i))(x) #Concatenate rows (c,w_i,ch,h)
        x = tfkl.Permute((3,4,1,2))(x) #ch,h,c,w_i
        x = tfkl.Reshape((ch,self.rows*h_i,self.cols*w_i))(x) #Concatenate cols (ch,h,w)
        x = tfkl.Permute((3,2,1))(x)

        return x

class MelScale(tfkl.Layer):
    def __init__(self,num_mel_bins=64,num_spectrogram_bins=None,sample_rate=None,lower_edge_hertz=125.0,upper_edge_hertz=3800.0,name=None, trainable=False): 
        super(MelScale, self).__init__(name=name)
        self.mel_args = {'mb':num_mel_bins, 
                         'sb':num_spectrogram_bins,
                         'sr': sample_rate,
                         'lh': lower_edge_hertz,
                         'uh': upper_edge_hertz}

    def call(self,x):
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.mel_args['mb'],
        num_spectrogram_bins=self.mel_args['sb'],
        sample_rate=self.mel_args['sr'],
        lower_edge_hertz=self.mel_args['lh'],
        upper_edge_hertz=self.mel_args['uh'])

        return tf.matmul(x,linear_to_mel_weight_matrix)

    def compute_output_shape(self,input_shape):
        num_mel_bins=self.mel_args['mb']
        output_shape = input_shape[:-1] + [num_mel_bins]
        return output_shape
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_mel_bins': self.mel_args['mb'],
            'num_spectrogram_bins': self.mel_args['sb'],
            'sample_rate': self.mel_args['sr'],
            'lower_edge_hertz': self.mel_args['lh'],
            'upper_edge_hertz': self.mel_args['uh']
        })
        return config

class OverlapAdd(tfkl.Layer):
    def __init__(self, hop_size=256, name=None, trainable=False):
        super(OverlapAdd, self).__init__(name=name)
        self.hop_size = hop_size

    def call(self,x):
        return tf.signal.overlap_and_add(x,self.hop_size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hop_size': self.hop_size,
        })
        return config

class SoftMask(tfkl.Layer):
    def __init__(self,name=None):
        super(SoftMask,self).__init__(name=name)

    def call(self,x):
        sources = x[0]
        to_mask = x[1]
        total_sources = tf.expand_dims(tf.reduce_sum(sources,axis=1),axis=1)
        mask = sources/(total_sources+1e-9)
        to_mask = tf.expand_dims(to_mask,axis=1)
        return mask*to_mask

class Spectrogram(tfkl.Layer):
    def __init__(self,win_size,hop_size,fft_size=None,calculate='magnitude',window=tf.signal.hann_window,pad_end=False,name=None, trainable=False):
        super(Spectrogram, self).__init__(name=name)

        self.stft_args = {'ws': win_size,
                  'hs': hop_size,
                  'ffts': fft_size,
                  'win': window,
                  'pad': pad_end,
                  'calculate': calculate}

    def call(self,x):
        stft = tf.signal.stft(
                signals=x,
                frame_length=self.stft_args['ws'],
                frame_step=self.stft_args['hs'],
                fft_length=self.stft_args['ffts'],
                window_fn=self.stft_args['win'],
                pad_end=self.stft_args['pad'])

        calculate = self.stft_args['calculate']
        if calculate == 'magnitude':
            return tf.abs(stft)
        elif calculate == 'complex':
            return stft
        elif calculate == 'phase':
            return tf.math.angle(stft)
        else:
            raise Exception("{} not recognized as calculate parameter".format(calculate))

    def compute_output_shape(self,input_shape):
        signal_len = input_shape[-1]
        f_bins = self.stft_args['ws']//2 + 1
        t_bins = np.floor((signal_len-self.stft_args['ws']+self.stft_args['hs'])/self.stft_args['hs']).astype(int)
        output_shape = input_shape[:-1] + [t_bins,f_bins]

        return output_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'win_size': self.stft_args['ws'],
            'hop_size': self.stft_args['hs'],
            'fft_size': self.stft_args['ffts'],
            'calculate': self.stft_args['calculate'],
            'window': self.stft_args['win'],
            'pad_end': self.stft_args['pad']
        })
        return config

class Window(tfkl.Layer):
    def __init__(self, window='hann', size=None, name=None, trainable=False):
        super(Window, self).__init__(name=name)
        self.window = window
        self.size = size

    def build(self, input_shape):
        if self.window == 'hann':
            self.window_array = tf.signal.hann_window(self.size)

    def call(self, x):
        return tf.multiply(x,self.window_array)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'window': self.window,
            'size': self.size
        })
        return config