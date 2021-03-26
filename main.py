from model import *
from data import *
from dice import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import json
#config
f = open('config/conf_unet.json', 'r')
config = json.load(f)
f.close()
startFold = config["startAtFold"]
nb_fold = config["kfold"]
batch_size = config["batch_size"]
# step_per_epoch = 2940
epoch = config["epochs"]
source_path = config["source_path"]
result_path = config["result_path"]
test_path = config["test_path"]
predict_path = config["predict_path"]
save_model_path = config["model_path"]
load_model = config["load_model"]
model_load = config["load_model_path"]
adapt_learning_rate = config["adapt_learning_rate"]
lr = config["learning_rate"]

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
data_den = dict()
for i in range(startFold,nb_fold) :
    myGene = trainGenerator(batch_size,source_path + 'Round' + str(i+1) + '/train/','img','label', data_den,image_color_mode='rgb',save_to_dir = None)
    valGene = trainGenerator(batch_size,source_path + 'Round' + str(i+1) + '/val/','img','label', data_den,image_color_mode='rgb',save_to_dir = None)
    if load_model :
        print("\n\n\ Load Model \n\n")
        model = unet(model_load)
    else :
        print("\n\n Not Load Model \n\n")
        model = unet()
        
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    if not os.path.exists(predict_path + 'Round' + str(i+1)):
        os.makedirs(predict_path + 'Round' + str(i+1))
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.exists(save_model_path + "val/"):
        os.makedirs(save_model_path + "val/")
        
    # calculate number of files in folders
    nb_train = len(os.listdir(source_path + 'Round' + str(i+1) + '/train/img/'))
    step_per_epoch_train = int(nb_train / batch_size)
    nb_val = len(os.listdir(source_path + 'Round' + str(i+1) + '/val/img/'))
    step_per_epoch_val = int(nb_val / batch_size)
    nb_test = len(os.listdir(test_path+'image/'))
    
    # Callback
    callbacks_list = []   
    model_checkpoint = ModelCheckpoint(save_model_path + str(i+1) + '_unet2.hdf5', monitor='loss',verbose=1, save_best_only=True)
    callbacks_list.append(model_checkpoint)
    val_checkpoint = ModelCheckpoint(save_model_path + 'val/' + str(i+1) + '_val-{epoch:03d}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    callbacks_list.append(val_checkpoint)
    tf_board = TensorBoard(log_dir='./Logs', 
                           histogram_freq=0, 
                           batch_size=batch_size, 
                           write_graph=True, 
                           write_grads=False, 
                           write_images=False, 
                           embeddings_freq=0, 
                           embeddings_layer_names=None, 
                           embeddings_metadata=None, 
                           embeddings_data=None, 
                           update_freq='epoch')
    callbacks_list.append(tf_board)

    if adapt_learning_rate:
            lrDecay = CyclicalLearningRateDecay(max_lr=lr, min_lr=1e-5,step_size=7)
            callbacks_list.append(LearningRateScheduler(lrDecay))
    print(f"\n\n Start Round ---> {str(i+1)} \n\n")
    
    # Train
    model.fit_generator(myGene,
                        steps_per_epoch=step_per_epoch_train,
                        epochs=epoch,
                        validation_data=valGene,
                        validation_steps=step_per_epoch_val,
                        callbacks=callbacks_list,
                        max_queue_size=20,
                        workers=1)

    #Validate
    valGene_test = testGenerator(source_path + 'Round' + str(i+1) + '/val/img/',num_image=nb_val,as_gray=False)
    results_val = model.predict_generator(valGene_test,nb_val,verbose=1)
    saveResult(predict_path + 'Round' + str(i+1) +'/result_val/',results_val,'val',i,False)
    val_score = calDice('val',i+1)
    print(f"Validate Dice Cofficient : {val_score}.")
    
    #Test
    testGene = testGenerator(test_path + 'image/',num_image=nb_test,as_gray=False)
    results_test = model.predict_generator(testGene,nb_test,verbose=1)
    saveResult(predict_path + 'Round' + str(i+1) +'/result_test/',results_test,'test',i,False)
    test_score = calDice('test',i+1)
    print(f"Testset Dice Cofficient : {test_score}.")
    
    # Calculate Dice Coefficient
    name_save = 'dice_'+ str(i+1) +'.txt'
            
    f = open(result_path + name_save, 'w+')
    f.write('Dice Coefficients\n')
    f.write('Validate Dice : ' + '{:.4f}'.format(val_score) + '\n\n')
    f.write('Test Dice : ' + '{:.4f}'.format(test_score) + '\n')
        
    f.close()