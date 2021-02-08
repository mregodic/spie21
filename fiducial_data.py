from __future__ import print_function

import os
import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave

from skimage.io import imread

data_path = './'

image_rows = int(352)
image_cols = int(352)
image_depth = int(224)


def test_data():
    
    data_path = '../production/'
    
    test_data_path = os.path.join(data_path, 'test/')
    dirs = os.listdir(test_data_path)
    
    total = len(dirs)
    
    imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
        
    for i in range(0, imgs.shape[0]):
        image_name = dirs[i]       
        print(os.path.join(test_data_path, image_name))
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)
        img = img.astype(np.uint8)
        img = np.array([img])
        imgs[i] = img
    
    print('Loading of test data done.')
 
    print('test mean=', str(np.mean(imgs)),' std=', str(np.std(imgs)))

    imgs = preprocess(imgs)
 
    print(imgs.shape)   

    return imgs

def train_data():
      
    data_path = '../production/'
    
    train_data_path = os.path.join(data_path, 'train/')
    mask_data_path = os.path.join(data_path, 'mask/')
    dirs = os.listdir(train_data_path)
    #total = int(len(dirs)*16*2)
    
    total = len(dirs)
    
    imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)

    #imgs_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint8)
    #imgs_mask_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    for i in range(0, imgs.shape[0]):
        image_name = dirs[i]       
        print(os.path.join(train_data_path, image_name))
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img = img.astype(np.uint8)
        img = np.array([img])
        imgs[i] = img
    
    print('Loading of train data done.')

    for i in range(0, imgs.shape[0]):
        mask_name = dirs[i]
        mask_name = mask_name.replace('_image', '_mask')
        print(os.path.join(mask_data_path, mask_name))
        img_mask = imread(os.path.join(mask_data_path, mask_name), as_grey=True)
        img_mask = img_mask.astype(np.uint8)
        img_mask = np.array([img_mask])
        imgs_mask[i] = img_mask
        

    print('Loading of masks done.')
    
    print('train mean=', str(np.mean(imgs)),' std=', str(np.std(imgs)))
    print('mask mean=', str(np.std(imgs_mask)),' std=', str(np.max(imgs_mask)))

    imgs_mask = preprocess(imgs_mask)
    imgs = preprocess(imgs)

    #print('Preprocessing of masks done.')

    #np.save('imgs_train.npy', imgs)
    #np.save('imgs_mask_train.npy', imgs_mask)

 #   imgs = preprocess_squeeze(imgs)
 #   imgs_mask = preprocess_squeeze(imgs_mask)

    print(imgs.shape)
    print(imgs_mask.shape)
   
    return imgs, imgs_mask


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    return imgs_test


def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=4)
    print(' ---------------- preprocessed -----------------')
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=4)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs

# check files
#train_data()
#test_data()