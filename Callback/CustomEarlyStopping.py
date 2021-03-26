#import keras lib
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, Callback

#import support lib
import numpy as np

class CustomEarlyStopping(EarlyStopping):
    
    def __init__(self, ratio=0.0,
                 patience=0, verbose=0, val_loss_name=None, loss_name=None):
        super(EarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.val_loss_name = 'val_loss' if val_loss_name is None else val_loss_name
        self.loss_name = 'loss' if loss_name is None else loss_name

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get(self.val_loss_name)
        current_train = logs.get(self.loss_name)
        if current_val is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(np.divide(current_train, current_val), self.ratio):
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))