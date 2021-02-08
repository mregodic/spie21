from __future__ import print_function

import os
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

np.random.seed(256)
import tensorflow as tf
#tf.set_random_seed(256)

from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model

from data3D import load_train_data, load_test_data, preprocess_squeeze

K.set_image_data_format('channels_last')

project_name = 'FiducalSegmentation'
img_rows = int(352)
img_cols = int(352)
img_depth = int(224)

smooth = 1.

from fiducial_data import train_data, test_data

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def create_model():
    inputs = Input((img_depth, img_rows, img_cols, 1), name='milonam')
    conv1 = Conv3D(8, (2, 2, 2), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(16, (2, 2, 2), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv5 = Conv3D(32, (2, 2, 2), activation='relu', padding='same')(pool2)

    up8 = concatenate([Conv3DTranspose(16 , (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
    conv8 = Conv3D(16, (2, 2, 2), activation='relu', padding='same')(up8)

    up9 = concatenate([Conv3DTranspose(8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    conv9 = Conv3D(8, (2, 2, 2), activation='relu', padding='same')(up9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def train():
    """
    Setting GPU to use.
    """
    # gpu_name = '0'
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = gpu_name
    # from keras.backend.tensorflow_backend import set_session
    # set_session(tf.Session(config=config))
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_mask_train = train_data()

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_train /= 255.  # scale masks to [0, 1]
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = create_model()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + '.h5'), monitor='val_loss', save_best_only=True, save_weights_only=False)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=70)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir,  project_name + '.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, imgs_mask_train, batch_size=1 , epochs=10000, verbose=1, shuffle=True, validation_split=0.25, callbacks=[model_checkpoint, csv_logger, early_stopping])

    print('-'*30)
    print('Training finished')
    print('-'*30)

def predict():

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
  
    imgs_test = test_data()

    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.  # scale masks to [0, 1]
    print('test shape', imgs_test.shape)

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model = create_model()
    weight_dir = 'weights'

    path = os.path.join(weight_dir, project_name + '.h5')

    model.load_weights(path)
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)

    #imgs_mask_test = imgs_test
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

    #imgs_mask_test = preprocess_squeeze(imgs_mask_test)
    #imgs_mask_test /= 1.7
    #imgs_mask_test = np.around(imgs_mask_test, decimals=0)
    imgs_mask_test = (imgs_mask_test*255.).astype(np.uint8)

    print('predicted mask shape', imgs_mask_test.shape)
    
    #imsave('J:/RhinospiderCT_UNET/production/predicted//pred_0.nrrd', imgs_mask_test[0], plugin='simpleitk')

    for i in range(0, imgs_mask_test.shape[0]):
        pred_img =  imgs_mask_test[i]
        
        dsc = dice_coef(pred_img.astype('float32'), imgs_test[0].astype('float32'))
        print('dice coeff = ', dsc)
        
        print('mask min=', str(np.min(pred_img)),' max=', str(np.max(pred_img)))
        imsave('../production/predicted//pred_' + str(i) + '.nrrd', pred_img, plugin='simpleitk')

    

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    
    #train()
    predict()


