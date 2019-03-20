"""
Train a UNET model to predict a continuous 3D image from a given
3D continuous brain image.

The example here uses the input image as a target image (aka an 'Autoencoder') but the
target image can be any other brain image.
"""

import operator
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras import callbacks as cbks
from keras import optimizers as opt
import nibabel as nib

base_dir = 'F:\\work\\programs\\gitProjects\\Unet-ants\\'
sys.path.insert(0, 'F:\\work\\programs\\gitProjects\\Unet-ants\\code\\')
os.chdir(base_dir+'code\\')
# local imports
from sampling import transforms as tx
from sampling import DataLoader, CSVDataset
from models import create_unet_model3D

#model_type = 'heart'
#model_type = 'brain'
model_type = 'abdominal'

data_dir = base_dir + 'data_3D/'
results_dir = base_dir+'results_3D/'
try:
    os.mkdir(results_dir)
except:
    pass

# tx.Compose lets you string together multiple transforms
co_tx = tx.Compose([tx.TypeCast('float32'),
                    tx.ExpandDims(axis=-1),
                    tx.RandomAffine(rotation_range=(-15, 15), # rotate btwn -15 & 15 degrees
                                    translation_range=(0.1,0.1), # translate btwn -10% and 10% horiz, -10% and 10% vert
                                    shear_range=(-10,10), # shear btwn -10 and 10 degrees
                                    zoom_range=(0.85,1.15), # between 15% zoom-in and 15% zoom-out
                                    turn_off_frequency=5,
                                    fill_value='min',
                                    target_fill_mode='constant',
                                    target_fill_value=0) # how often to just turn off random affine transform (units=#samples)
                    ])

input_tx = tx.MinMaxScaler((-1,1)) # scale between -1 and 1

target_tx = tx.OneHot() # convert segmentation image to One-Hot representation for cross-entropy loss

# use a co-transform, meaning the same transform will be applied to input+target images at the same time 
# this is necessary since Affine transforms have random parameter draws which need to be shared

if model_type == 'heart':
    csv_filname = 'heart_train.csv'

elif model_type == 'brain':
    csv_filname = 'image_filemap_quarter.csv'

elif model_type == 'abdominal':
    csv_filname = 'abdominal_train.csv'
else:
    print('wrong model type!')
    exit(0)

dataset = CSVDataset(filepath=data_dir+csv_filname,
                    base_path=os.path.join(data_dir,'images'), # this path will be appended to all of the filenames in the csv file
                    input_cols=['Images'], # column in dataframe corresponding to inputs (can be an integer also)
                    target_cols=['Segmentations'],# column in dataframe corresponding to targets (can be an integer also)
                    input_transform=input_tx, target_transform=target_tx, co_transform=co_tx,
                    co_transforms_first=True) # run co transforms before input/target transforms

# split into train and test set based on the `train-test` column in the csv file
# this splits alphabetically by values, and since 'test' comes before 'train' thus val_data is returned before train_data
val_data, train_data = dataset.split_by_column('TrainTest')


# print info on input data
for i in range(len(train_data)):
    print('num of labels in ' + str(i) + " train dataset: " + str(train_data[i][1].shape))

for i in range(len(val_data)):
    print('num of labels in ' + str(i) + " validation dataset: " + str(val_data[i][1].shape))
print('val_data = '+str(len(val_data)))
print('train_data = '+str(len(train_data)))


# overwrite co-transform on validation data so it doesnt have any random augmentation
# overwrite co-transform on validation data so it doesnt have any random augmentation
val_data.set_co_transform(tx.Compose([tx.TypeCast('float32'),
                                      tx.ExpandDims(axis=-1)]))

# create a dataloader .. this is basically a keras DataGenerator -> can be fed to `fit_generator`
batch_size = 1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# write an example batch to a folder as JPEG
#train_loader.write_a_batch(data_dir+'example_batch/')

n_labels = train_data[0][1].shape[-1]
print('train_data shape = '+str(train_data[0][0].shape))
# create model
model = create_unet_model3D(input_image_size=train_data[0][0].shape, n_labels=n_labels, layers=4,
                            mode='classification')


callbacks = [cbks.ModelCheckpoint(results_dir+'segmentation-weights.h5', monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]



model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size),
                    epochs=200, verbose=1, callbacks=callbacks,
                    shuffle=True,
                    validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size),
                    class_weight=None, max_queue_size=10,
                    workers=1, use_multiprocessing=False,  initial_epoch=0)




# evaluate the model
real_val_x, real_val_y = val_data.load()
scores = model.evaluate(real_val_x, real_val_y, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# serialize  model to JSON, weights to HDF5
model_json = model.to_json()


if model_type == 'heart':
    with open("..\\results_3D\\model_heart.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("..\\results_3D\\model_heart_weights.h5")

elif model_type == 'brain':
    with open("..\\results_3D\\model_brain.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("..\\results_3D\\model_brain_weights.h5")

elif model_type == 'abdominal':
    with open("..\\results_3D\\model_abdominal.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("..\\results_3D\\model_abdominal_weights.h5")
else:
    print('wrong model type!')
    exit(0)

print("Saved model to disk")


# check trained model on the validation data and print results for the first validation dataset
real_val_y_pred = model.predict(real_val_x)
sh = real_val_y_pred.shape[1:4]

# fill numpy arrays for predicted segmentation and input anatomy
pred_segm = np.ndarray(shape=sh, dtype=np.int32)
anatomy = np.ndarray(shape=sh, dtype= np.int32)
for z in range(sh[0]):
    for y in range(sh[1]):
        for x in range(sh[2]):
            index, value = max(enumerate(real_val_y_pred[0][z][y][x]), key=operator.itemgetter(1))
            a = real_val_y_pred[0,z,y,x]
            #print (real_val_y_pred[0][z][y][x])

            pred_segm[z][y][x] = index
            anatomy[z,y,x] = (1+real_val_x[0, z, y, x,0]) * 1000

# save predicted segmentation
img = nib.Nifti1Image(pred_segm, np.eye(4))
nib.save(img, os.path.join('build', '..\\..\\predicted_segm.nii.gz'))

# save input anatomy
ai = nib.Nifti1Image(anatomy, np.eye(4))
nib.save(ai, os.path.join('build', '..\\..\\input_anatomy.nii.gz'))
