#import support lib
import json
import os
from PIL import Image, ImageOps
import pydicom
import numpy as np
import shutil
from collections import OrderedDict

#import local lib
from Database.Sqlite3 import DB_SQLITE3
from Preprocess.PreprocessImage import PreprocessImage
from Preprocess.AugmentationImage import AugmentationImage
from Preprocess.PreprocessArray import PreprocessArray

class SupportLibrary():
    
    def __init__(self, db_instance):
        
        f = open('config/conf_training.json', 'r')
        config = json.load(f)
        f.close()
        
        self.dataset_table = config['table_dataset_information']
        self.label_table = config['table_label_information']
        self.dataset_path = config['dataset_path']
        self.augmentation_list = config['augmentation_list']
        self.augmentation_all_class = config['augmentation_all_class']
        self.augmentation_class_list = config['augmentation_class_list']
        
        self.db = db_instance
        
    def dicomToPng(self, dicom_path, png_path):
        
        #Extracting data from the mri file
        plan = pydicom.read_file(dicom_path)
        shape = plan.pixel_array.shape

        #Convert to float to avoid overflow or underflow losses.
        image_2d = plan.pixel_array.astype(float)

        #Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    
        #Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)

        #Writing the PNG file
        w = Image.fromarray(image_2d_scaled)
        w.save(png_path, 'png')
    
    def dicomToPng_dir(self, dicom_dir, png_dir):
        
        #Create the folder for the pnd directory structure
        if os.path.exists(png_dir):
            shutil.rmtree(png_dir)
        os.makedirs(png_dir)

        #Recursively traverse all sub-folders in the path
        for dicom_sub_folder, subdirs, files in os.walk(dicom_dir):
            for dicom_file in os.listdir(dicom_sub_folder):
                dicom_file_path = os.path.join(dicom_sub_folder, dicom_file)

                # Make sure path is an actual file
                if os.path.isfile(dicom_file_path):

                    # Replicate the original file structure
                    rel_path = os.path.relpath(dicom_sub_folder, dicom_dir)
                    png_folder_path = os.path.join(png_dir, rel_path)
                    if not os.path.exists(png_folder_path):
                        os.makedirs(png_folder_path)
                    png_file_path = os.path.join(png_folder_path, '%s.png' % '.'.join(dicom_file.split('.')[0:-1]))

                    try:
                        # Convert the actual file
                        self.dicomToPng(dicom_file_path, png_file_path)
                        print('SUCCESS: %s --> %s' % (dicom_file_path, png_file_path))
                    except Exception as e:
                        print('FAIL: %s --> %s : %s' % (dicom_file_path, png_file_path, e))
                        
    def update_datapath(self):
        
        count_rows = self.db.count(self.dataset_table)
        count = 0
        
        for dataset_sub_folder, subdirs, files in os.walk(self.dataset_path):
                
            for dataset_file in os.listdir(dataset_sub_folder):
                
                rel_path = os.path.join(dataset_sub_folder, dataset_file)
                
                # Make sure path is an actual file
                if not os.path.isfile(rel_path):
                    continue
                
                row = self.db.select_by_cond(self.dataset_table, ['id'], {'path':'%' + dataset_file + '%'}, ['like'])[0]
                
                if row == None:
                    print(dataset_file + 'not found')
                    continue
                
                self.db.update_by_id(self.dataset_table, row[0], {'path':rel_path})
                count += 1
                print('%.4f %%' % ((count/count_rows)*100.0), end='\r')
        
        self.db.commit() # commit for update database
            
            
    def create_dataset_db(self, drop_exist=False):
        
        dict_fields = OrderedDict([('path', 'text'), ('type','text'), ('using_type','text')])
        
        self.db.create_table(self.dataset_table, dict_fields, drop_exist)
        
        classes_list = sorted(os.listdir(self.dataset_path))
        
        for each_class in classes_list:
            
            each_class_path = self.dataset_path + '/' + each_class
            
            for dataset_sub_folder, subdirs, files in os.walk(each_class_path):
                
                for dataset_file in os.listdir(dataset_sub_folder):
                    
                    rel_path = os.path.join(dataset_sub_folder, dataset_file)
                    
                    # Make sure path is an actual file
                    if not os.path.isfile(rel_path):
                        continue
                    
                    self.db.insert(self.dataset_table, {'path':rel_path, 'type':each_class, 'using_type':'train'})
                    
        
        self.db.commit() # commit for insert to database
                    
    
    def create_label_table(self, drop_exist=False, splitor_for_multiclass='|'):
        
        if not self.db.has_table(self.dataset_table):
            
            raise NotImplementedError("Must implement dataset table before call this method")
        
        
        dict_fields = {'label':'text'}
        self.db.create_table(self.label_table, dict_fields, drop_exist)
        
        count_rows = self.db.count(self.dataset_table)
        #label_added = self.db.select_by_cond(self.label_table) # for increse performance
        #label_added = list(label_added) if label_added != None else []
        label_added = []
        
        for id in range(1, count_rows + 1):
            
            row = self.db.select_by_id(self.dataset_table, id)
            
            if row == None:
                continue
                
            labels = row[2].split(splitor_for_multiclass)
            
            for label in labels:
                
                if not label in label_added:
                    # Make sure this label not found in table
                    row_label = self.db.select_by_cond(self.label_table, ['label'], {'label':label})
                
                    if not row_label:
                        self.db.insert(self.label_table, {'label':label})
                        label_added.append(label)
                
            print('%.4f %%' % ((id/count_rows)*100.0), end='\r')
        
        self.db.commit() # commit for insert to database
        
    
    def preprocess_data(self, img, preprocess_list):
        
        pre_img = PreprocessImage()
        
        for preprocess_method in preprocess_list:
            img = pre_img.preprocess(img, preprocess_method)
        
        return img
    
    
    def preprocess_data_array(self, img, preprocess_list):
        
        pre_arr = PreprocessArray()
        
        for preprocess_method in preprocess_list:
        
            img = pre_arr.preprocess(img, preprocess_method)
        
        return img
    
    
    def data_augmentation(self, img, current_augmentation):
        
        augmentator = AugmentationImage(self.augmentation_list)
        img_aug = augmentator.data_augmentation(img, current_augmentation)
        
        return img_aug
    
    
    def calculate_augmentation_part(self):
        
        augmentator = AugmentationImage(self.augmentation_list)
        return augmentator.calculate_augmentation_part()
    
    
    def check_augmentation(self, is_augmentation, labels):
        
        if not is_augmentation:
            return False
        
        if self.augmentation_all_class:
            return True
        
        for label in labels:
            if str(label) in self.augmentation_class_list:
                return True
        
        return False
        
        
    def generate_multiple_input(self, img, index_multiple):
        
        img = img.convert('RGB')
        
        return ImageOps.invert(img)
        
        
if __name__ == '__main__':
    
    db = DB_SQLITE3('./config/deep_xray_udon.db')
    prep = SupportLibrary(db)
    #prep.update_datapath()
    #prep.create_dataset_db(True)
    #prep.create_label_table(True)
            