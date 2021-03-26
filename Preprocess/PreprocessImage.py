#import support library
import numpy as np
import cv2
from PIL import Image, ImageOps
from threading import Thread
import json
import math

class PreprocessImage():
    
    def __init__(self):
        
        f = open('config/conf_training.json', 'r')
        config = json.load(f)
        f.close()
            
        self.target_size = (config['image_size'][0], config['image_size'][1])
        
        
    def preprocess(self, img_pil, preprocess_method):
        
        if preprocess_method == 'clahe':
            img_pil = self.clahe(img_pil)
            
        elif preprocess_method == 'clahe_r':
            img_pil = self.clahe(img_pil, 0)
            
        elif preprocess_method == 'clahe_g':
            img_pil = self.clahe(img_pil, 1)
            
        elif preprocess_method == 'clahe_b':
            img_pil = self.clahe(img_pil, 2)
            
        elif preprocess_method == 'invert':
            img_pil = self.invert(img_pil)
            
        elif preprocess_method == 'RGB':
            img_pil = self.toRGB(img_pil)
            
        elif preprocess_method == 'resize':
            img_pil = self.resize(img_pil)
            
        elif preprocess_method == 'padding_reflect':
            img_pil = self.padding_reflect(img_pil)
        
        return img_pil
    
    def padding_reflect(self, img_pil):
        
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        shape = img.shape
        width = shape[1]
        height = shape[0]
        
        dif = width - height
        padding_length = abs(dif) // 2
        
        if (dif > 0):
            img = cv2.copyMakeBorder(img, padding_length, padding_length, 0, 0, cv2.BORDER_REFLECT)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, padding_length, padding_length, cv2.BORDER_REFLECT)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        return img
        
    def clahe(self, img_pil,ch=None):
        
        dict_result = {}
        threads = {}
        
        # create a CLAHE object (Arguments are optional).
        img_ori = np.array(img_pil)
    
        for i in range(3):
            if ch != None:
                if ch != i:
                    continue
            img = img_ori[:,:,i]
            threads[i] = Thread(target=self.__clahe_process__, args=(img, dict_result, i))
            threads[i].start()
            
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             cl1 = clahe.apply(img)
#             img_ori[:,:,i] = np.array(cl1)

        for i in threads:
            threads[i].join()
            
        for i in dict_result:
            img_ori[:, :, i] = dict_result[i]
        
        img = Image.fromarray(np.array(img_ori))
        
        return img
    
    
    def __clahe_process__(self, img_gray, dict_result, key):
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img_gray)
        
        dict_result[key] = np.array(cl1)
    
    
    def invert(self, img_pil):
        
        return ImageOps.invert(img_pil)
    
    def resize(self, img_pil):
        
        return img_pil.resize(self.target_size)
    
    
    def toRGB(self, img_pil):
        
        return img_pil.convert('RGB')
    
    def gadiant(self, img_pill):
        
        img = np.array(img_pil)
        shape = img.shape
        width = shape[1]
        height = shape[0]
        vlin = np.repeat(np.expand_dims(np.linspace(0, 255, height).astype('uint8'),axis=-1), width, axis=1)
        
        return vlin 