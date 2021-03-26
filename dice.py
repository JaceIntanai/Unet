from PIL import Image
import numpy as np
import skimage.transform as trans
from glob import glob
import json

f = open('config/conf_unet.json', 'r')
config = json.load(f)
f.close()
target_size = config["image_size"]
source_path = config["source_path"]
test_path = config["test_path"]
result_path = config["result_path"]
predict_path = config["predict_path"]

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def calDice(locate, r) :
    
    if(locate == 'val') :
        answerPath = source_path + 'Round' + str(r) + '/val/label/'
        predictPath = predict_path + 'Round' + str(r) + '/result_val/'
    elif(locate == 'test') :
        answerPath = test_path + 'label/'
        predictPath = predict_path + 'Round' + str(r) + '/result_test/'
    else :
        raise ValueError("Error Dice Coefficient.")
    
    answer = glob(answerPath + '*.png')
#     print(len(answer))
    
    sum = 0
    for t in answer:
        # ground truth image
        gt  = Image.open(t).convert('L')
        gt = np.array(gt)
        gt = trans.resize(gt,target_size)

        imgName = t[t.find("/T1") + 1 : -4]

        # segment image from model
        seg  = Image.open(predictPath +imgName+ '.png')
        seg = np.array(seg)

        sum = sum + dice(gt,seg)
        
    dice_score = sum / len(answer)
#         print(dice_score)
        
    return dice_score