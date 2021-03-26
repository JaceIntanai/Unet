from model import *
from data import *
import os 
#config
load_weight = 'Result/Unet1_v3/model_output/1_unet2.hdf5'
testPath = 'resource/dataset_ev/image/'
# testPath = 'Check/image/'
test_list = os.listdir(testPath)
nb_test = len(test_list)
# savePath = 'Result/Unet2_lvl9_clahe_r/predict_normal/'
# savePath = 'resource/predict_unet2_v2/'
savePath = 'Images/Unet1_v3/'

# print(test_list)

model = unet()
model.load_weights(load_weight)

testGene = testGenerator(testPath,num_image=nb_test,as_gray=False)
results = model.predict_generator(testGene,nb_test,verbose=1)
# print(results)


saveResult(savePath,results,'test',0,True)