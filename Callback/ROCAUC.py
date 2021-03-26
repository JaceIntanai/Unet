from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from Generator import Generator
import numpy as np

class ROCAUC(Callback):
    
    def __init__(self, val_id, val_step, nb_line_plot, db=None, augmentation=False, is_multiple_input=False, multiple_input=2, is_ensemble=False):
        
        self.db = db
        self.val_id = val_id
        self.val_step = val_step
        self.nb_line_plot = nb_line_plot
        self.augmentation = augmentation
        self.is_multiple_input = is_multiple_input
        self.multiple_input = multiple_input
        self.is_ensemble = is_ensemble
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        
        if self.db is None:
            return
        
        val_generator = Generator(self.db)
        
        roc_flow = val_generator.flow_db_multiple_input(self.val_id, self.multiple_input, 
                            augmentation=self.augmentation) if self.is_multiple_input else val_generator.flow_db(self.val_id, augmentation=self.augmentation)
        
        y_pred = self.model.predict_generator(roc_flow, steps=self.val_step)
        
        if self.is_ensemble:
            y_pred = ROCAUC.__reformat_from_ensemble_predict__(y_pred)
            
        y = np.array(val_generator.classes)
        
        for i in range(self.nb_line_plot):
            roc = roc_auc_score(y[:, i], y_pred[:, i])
            print('\rroc-auc: %s' % (str(round(roc,4))) ,end=100*' '+'\n')

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
    @staticmethod
    def __reformat_from_ensemble_predict__(preds):
        
        preds_output = []
        
        for i in range(len(preds)):    
            for j in range(len(preds[i])):
                
                if i == 0:
                    preds_output.append([])
                
                preds_output[j].append(preds[i][j][0])
                
        return np.array(preds_output)