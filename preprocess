import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import glob
import random as r
import skimage.io as io
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Input, UpSampling2D,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from keras.models import load_model


img_size = 120

def train_array(path,end,label=False, resize=(155,img_size,img_size)): #flair,t2 için
    files = glob.glob(path+end,recursive=True)
    img_liste = []
    r.seed(9)
    r.shuffle(files)
    for file in files:
        img = io.imread(file,plugin='simpleitk')
        img = (img-img.mean())/ img.std()
        img.astype('float32')
        
        for slice in range(50,130): #kesit aralığı
            img_s = img[slice,:,:]
            img_s = np.expand_dims(img_s,axis=1)
            img_liste.append(img_s)
            
    name = 'x_'+ str(img_size)
    np.save(name, np.array(img_liste),np.float32)
    print('Saved', len(files), 'to', name)
    return np.array(img_liste,np.float32)

def seg_array(path,end,label=True, resize=(155,img_size,img_size)): #seg data için
    files = glob.glob(path+end,recursive=True)
    img_liste = []
    r.seed(9)
    r.shuffle(files)
    for file in files:
        img = io.imread(file,plugin='simpleitk')
        
        if label == 1:
            img[img != 0 ] = 1 # tam tümör
        if label == 2:
            img[img != 1 ] = 0 # nekroz
        if label == 3:
            img[img == 2 ] = 0 # ödemsiz tümör
            img[img != 0 ] = 1
        if label == 4:
            img[img != 4 ] = 0 # genişleyen tümör
            img[img == 4 ] = 1
         
        else:
            img = (img-img.mean()) / img.std()      # flair images
            
        
        for slice in range(50,130): #kesit aralığı
            img_s = img[slice,:,:]
            img_s = np.expand_dims(img_s,axis=0)
            img_liste.append(img_s)
            
    name = 'y_'+ str(img_size)
    np.save(name, np.array(img_liste),np.float32)
    print('Saved', len(files), 'to', name)
    return np.array(img_liste,np.float32)

def aug(scans,n):          
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=25,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=False)
    i=0
    img_liste=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=1000):
        img_liste=np.vstack([img_liste,batch])
        i += 1
        if i == n:
            break
    return img_liste

#verileri okuma
flair = train_array('D:/Kullanıcılar/.spyder-py3/bitirme/MICCAI_BraTS2020_TrainingData', '**\\*_flair.nii',label=False, resize=(155,img_size,img_size))
seg = seg_array('D:/Kullanıcılar/.spyder-py3/bitirme/MICCAI_BraTS2020_TrainingData', '**\\*_seg.nii',label=True, resize=(155,img_size,img_size)) #sondaki değer label değeri
print(flair.shape)

K.set_image_data_format('channels_first') # (240,240,1) => (1,240,240) yani katman sayisi ilk

