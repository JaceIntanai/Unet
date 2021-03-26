#import support library
import numpy as np
import cv2
from threading import Thread

class PreprocessArray():
    
    def __init__(self):
        
        x = 5
        
    def preprocess(self, img, preprocess_method):
        
        if preprocess_method == '01normalization':
            img = self.normalization01(img)
            
        elif preprocess_method == 'log':
            img = self.log(img)
            
        elif preprocess_method == 'dct':
            img = self.dct(img)
        
        return img
        
        
    def normalization01(self, img):
        
        img = img / 255.0
        
        return img
    
    def log(self, img):
        
        img = np.log(img)
        
        return img
    
    def dct(self, img, ch=None):
        
        dict_result = {}
        threads = {}
        
        for i in range(3):
            if ch != None:
                if ch != i:
                    continue
            img_tmp = img[:,:,i]
            threads[i] = Thread(target=self.__dct_process__, args=(img_tmp, dict_result, i))
            threads[i].start()
#             img_tmp = np.float32(img_tmp) / 255.0
#             img_tmp = cv2.dct(img_tmp)
#             img[:,:,i] = np.array(img_tmp)

        for i in threads:
            threads[i].join()
            
        for i in dict_result:
            img[:, :, i] = dict_result[i]
        
        return img
    
    
    def __dct_process__(self, img_gray, dict_result, key):
        
        img_gray = np.float32(img_gray) / 255.0
        img_gray = cv2.dct(img_gray)
        
        dict_result[key] = img_gray
    
    