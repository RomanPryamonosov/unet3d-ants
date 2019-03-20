"""
Train a UNET model to predict a continuous 3D image from a given
3D continuous brain image.

The example here uses the input image as a target image (aka an 'Autoencoder') but the
target image can be any other brain image.
"""

import tensorflow as tf
from keras.utils.np_utils import to_categorical
import operator
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.losses import mean_squared_error
from keras import callbacks as cbks
import nibabel as nib
import keras.backend as K


def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

model_type = 'heart'
#model_type = 'brain'
#model_type = 'abdominal'

def tversky_loss(y_true, y_pred, alpha=1.0/np.log(1.03+0.994), beta=1.0/np.log(1.036), smooth=1e-10):
    """ Tversky loss function.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer

base_dir = 'F:\\work\\programs\\gitProjects\\Unet-ants\\'
sys.path.insert(0, 'F:\\work\\programs\\gitProjects\\Unet-ants\\code\\')
os.chdir(base_dir+'code\\')
# local imports
from sampling import transforms as tx
from sampling import DataLoader, CSVDataset
from models import create_unet_model3D


data_dir = base_dir + 'data_3D/'
results_dir = base_dir+'results_3D/'
try:
    os.mkdir(results_dir)
except:
    pass


input_tx = tx.MinMaxScaler((-1,1)) # scale between -1 and 1

target_tx = tx.OneHot() # convert segmentation image to One-Hot representation for cross-entropy loss

# load json and create model
print("Loading model from disk")
if model_type == 'heart':
    json_file = open('..\\results_3D\\model_heart.json', 'r')
elif model_type == 'brain':
    json_file = open('..\\results_3D\\model_brain.json', 'r')
elif model_type == 'abdominal':
    json_file = open('..\\results_3D\\model_abdominal.json', 'r')
else:
    print('wrong model type!')
    exit(0)

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

print("Loading model weights from disk")
# load weights into new model
if model_type == 'heart':
    loaded_model.load_weights("..\\results_3D\\model_heart_weights.h5")
elif model_type == 'brain':
    loaded_model.load_weights("..\\results_3D\\model_brain_weights.h5")
elif model_type == 'abdominal':
    loaded_model.load_weights("..\\results_3D\\model_abdominal_weights.h5")
else:
    print('wrong model type!')
    exit(0)

print('Done!')
print('Now loading a test dataset')

# evaluate loaded model on test data
# loaded_model.compile(loss=tversky_loss, optimizer='rmsprop', metrics=[tversky_loss])
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_crossentropy'])


if model_type == 'heart':
    test_data_nii = nib.load(os.path.join(data_dir, 'images\\heart_test\\anat1_q.nii.gz'))
    true_segm_nii = nib.load(os.path.join(data_dir, 'images\\heart_test\\segm1_q.nii.gz'))
elif model_type == 'brain':
    test_data_nii = nib.load(os.path.join(data_dir, 'images\\quarter\\anat14_q.nii.gz'))
    true_segm_nii = nib.load(os.path.join(data_dir, 'images\\quarter\\segm14_q.nii.gz'))
elif model_type == 'abdominal':
    test_data_nii = nib.load(os.path.join(data_dir, 'images\\abdominal\\resampled\\anat11.nii.gz'))
    true_segm_nii = nib.load(os.path.join(data_dir, 'images\\abdominal\\resampled\\segm11.nii.gz'))
else:
    print('wrong model type!')
    exit(0)

original_header = test_data_nii.header
test_data = test_data_nii.get_data()
print("test_data = " + str(test_data.shape))
nib.save(test_data_nii, os.path.join('build', '..\\..\\anatomy.nii.gz'))

true_segm = true_segm_nii.get_data()
print("true_segm = " + str(true_segm.shape))
nib.save(true_segm_nii, os.path.join('build', '..\\..\\true_segm.nii.gz'))



# expand dimentions of test data and segmentation data
# also rescale test data to be in (-1,1)
real_val_x = input_tx.transform(np.expand_dims(np.expand_dims(test_data, 0), -1))
real_val_y = np.expand_dims(to_categorical(true_segm), 0)


# predict on a test dataset
real_val_y_pred = loaded_model.predict(real_val_x)


# monitor score
score = loaded_model.evaluate(real_val_x, real_val_y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

print('predicted!')


# Write predicted data
sh = real_val_y_pred.shape[1:4]
#print(sh)
pred_segm = np.ndarray(shape=sh, dtype=np.int32)
#true_segm = np.ndarray(shape=sh, dtype=np.int32)
anatomy = np.ndarray(shape=sh, dtype=np.int32)

for z in range(sh[0]):
    for y in range(sh[1]):
        for x in range(sh[2]):
            index, value = max(enumerate(real_val_y_pred[0][z][y][x]), key=operator.itemgetter(1))
            #print (real_val_y_pred[0][z][y][x])
            pred_segm[z][y][x] = index
#            index, value = max(enumerate(real_val_y[0][z][y][x]), key=operator.itemgetter(1))
#            true_segm[z][y][x] = index
            anatomy[z,y,x] = (1+real_val_x[0,z,y,x,0]) * 1000
img = nib.Nifti1Image(pred_segm, np.eye(4))
img.header['pixdim'] = original_header['pixdim']
nib.save(img, os.path.join('build', '..\\..\\predicted_segm.nii.gz'))

anatomy_img = nib.Nifti1Image(anatomy, np.eye(4))
#anatomy_img.header['pixdim'][1:4] = original_spacing
anatomy_img.header['pixdim'] = original_header['pixdim']
anatomy_img.header['qoffset_x'] = original_header['qoffset_x']
anatomy_img.header['qoffset_y'] = original_header['qoffset_y']
anatomy_img.header['qoffset_z'] = original_header['qoffset_z']
anatomy_img.header['xyzt_units'] = original_header['xyzt_units']
anatomy_img.header['quatern_b'] = original_header['quatern_b']
anatomy_img.header['quatern_c'] = original_header['quatern_c']
anatomy_img.header['quatern_d'] = original_header['quatern_d']


nib.save(anatomy_img, os.path.join('build', '..\\..\\input_anatomy.nii.gz'))

