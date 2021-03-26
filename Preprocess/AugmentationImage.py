#import support lib
import json
import numpy as np
import cv2
from PIL import ImageOps, Image

class AugmentationImage():
    
    def __init__(self, augmentation_list = ['rotation', 'invert', 'horizontal_flip', 'vertical_flip']):
        
        f = open('config/conf_training.json', 'r')
        config = json.load(f)
        
        self.nb_rotation = config['nb_rotation']
        
        self.augmentation_list = augmentation_list
        
    def data_augmentation(self, img, current_augmentation):
        
        augmentation_range = 1
        
        if current_augmentation == 0:
            
            return img
        
        if 'rotation' in self.augmentation_list:
            
            prev_index = augmentation_range
            augmentation_range += self.__count_rotation__()
            
            if current_augmentation < augmentation_range:
                # print only once
                if current_augmentation == prev_index :
                    print('rotation')
        
                return self.__rotation__(img, current_augmentation)
        
        if 'invert' in self.augmentation_list:
        
            augmentation_range += self.__count_invert__()
        
            if current_augmentation < augmentation_range:
            
                return self.__invert__(img)
            
        if 'horizontal_flip' in self.augmentation_list:
            
            prev_index = augmentation_range
            augmentation_range += self.__count_horizontal_flip__()
        
            if current_augmentation < augmentation_range:
                # print only once
                if current_augmentation == prev_index + 1:
                    print('horizontal_flip')
            
                return self.__horizontal_flip__(img)
        
        if 'vertical_flip' in self.augmentation_list:
            
            prev_index = augmentation_range
            augmentation_range += self.__count_vertical_flip__()
        
            if current_augmentation < augmentation_range:
                # print only once
                if current_augmentation == prev_index + 1:
                    print('vertical_flip')
            
                return self.__vertical_flip__(img)
            
        if 'h_flip_rotation' in self.augmentation_list:
            
            prev_index = augmentation_range
            augmentation_range += self.__count_h_flip_rotation()
        
            if current_augmentation < augmentation_range:
                # print only once
                if current_augmentation == prev_index + 1:
                    print('h_flip_rotation')
            
                return self.__h_flip_rotation__(img, current_augmentation-prev_index)
            
        if 'v_flip_rotation' in self.augmentation_list:
            
            prev_index = augmentation_range
            augmentation_range += self.__count_v_flip_rotation()
        
            if current_augmentation < augmentation_range:
                # print only once
                if current_augmentation == prev_index + 1:
                    print('v_flip_rotation')
            
                return self.__v_flip_rotation__(img, current_augmentation-prev_index)
            
        if 'padding_reflect' in self.augmentation_list:
            
            augmentation_range += self.__count_padding_reflect__()
        
            if current_augmentation < augmentation_range:
            
                return self.__padding_reflect__(img)
        
        return img
        
    
    def calculate_augmentation_part(self):
        
        count = 1
        
        if 'rotation' in self.augmentation_list:
            count += self.__count_rotation__()
            
        if 'invert' in self.augmentation_list:
            count += self.__count_invert__()
            
        if 'horizontal_flip' in self.augmentation_list:
            count += self.__count_horizontal_flip__()
            
        if 'vertical_flip' in self.augmentation_list:
            count += self.__count_vertical_flip__()
            
        if 'padding_reflect' in self.augmentation_list:
            count += self.__count_padding_reflect__()
            
        if 'h_flip_rotation' in self.augmentation_list:
            count += self.__count_h_flip_rotation()
        
        if 'v_flip_rotation' in self.augmentation_list:
            count += self.__count_v_flip_rotation()
        
        return count
    
    
    def __rotation__(self, img, current_augmentation):
        
        count = 1
        rotation_angle = int(360/self.nb_rotation)
        
        for i in range(1, 360):
            
            if i % rotation_angle != 0:
                continue
            
            if count == current_augmentation:
                img = img.rotate(i)
                return img
            
            count += 1
            
        return img
    
    def __invert__(self, img):
        
        return ImageOps.invert(img)
    
    def __horizontal_flip__(self, img):
        
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    def __vertical_flip__(self, img):
        
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    def __h_flip_rotation__(self, img, current_augmentation):
        
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.__rotation__(img, current_augmentation)
    
    def __v_flip_rotation__(self, img, current_augmentation):
        
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return self.__rotation__(img, current_augmentation)
    
    def __padding_reflect__(self, img_pil):
        
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
    
    
    def __count_rotation__(self):
        
        count = 0
        rotation_angle = int(360/self.nb_rotation)
        
        for i in range(1, 360):
            if i % rotation_angle != 0:
                continue
            
            count += 1
        
        return count
    
    def __count_invert__(self):
        
        return 1
    
    def __count_horizontal_flip__(self):
        
        return 1
    
    def __count_vertical_flip__(self):
        
        return 1
    
    def __count_padding_reflect__(self):
        
        return 1
    
    def __count_h_flip_rotation(self):
        
        return self.__count_rotation__()
    
    def __count_v_flip_rotation(self):
        
        return self.__count_rotation__()
    

if __name__ == '__main__':
    
    augmentator = AugmentationImage()
    img = Image.open('try.png')
    img = augmentator.__invert__(img)
    img.save('result.png')