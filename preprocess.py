import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import skimage.io as io
import skimage.transform as trans
import SimpleITK as sitk

#bu islem veri büyütmenin modelin eğitim snuçlarında karşılaştırma yapmak için yapılmıştır.

# mri görüntülerin boyutu
img_size = 120 


# arttırma sayısı
num_of_aug = 1

#total örneklem
total_examples = 8000

# eğitim görüntülerine önişleme uygulama
def data_array(path, end, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(path + end, recursive=True)
    imgs = []
    print('Processing---', end)
    
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        img = trans.resize(img, resize, mode='constant')
        
        
        if label == 1:                          #seg görüntülerinde piksel eşitleme
            img[img != 0 ] = 1 # tam tümör
        if label == 2:
            img[img != 1 ] = 0 # nekroz
        if label == 3:
            img[img == 2 ] = 0 # ödemsiz tümör
            img[img != 0 ] = 1
        if label == 4:
            img[img != 4 ] = 0 # genişleyen tümör
            img[img == 4 ] = 1
         
            img = img.astype('float32')
            
        else:
            img = (img-img.mean()) / img.std()      # flair görüntüleri için normalizasyon işlemi
            
        for slice in range(50,130):
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)
            img_g = aug(img_t,num_of_aug)       #arttırma işlemi
            
            for n in range(img_g.shape[0]):
                imgs.append(img_g[n,:,:,:])
                
    name = 'y_'+ str(img_size) if label else 'x_'+ str(img_size)    #array kaydedilir
    np.save(name, np.array(imgs).astype('float32'))
    print('Saved', len(files), 'to', name)


def aug(scans,n):          #giriş görüntülerinin boyutu 4 olmalıdır
    datagen = ImageDataGenerator(   #arttırma fonksiyonu parametreleri
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
    img_g=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=1000):
        img_g=np.vstack([img_g,batch])
        i += 1
        if i == n:
            break
    return img_g


flair = data_array('D:\\Dataset\\MICCAI_BraTS2020_TrainingData\\', '**\\*_flair.nii.gz',label=False, resize=(155,img_size,img_size))
seg = data_array('D:\\Dataset\\MICCAI_BraTS2020_TrainingData\\', '**\\*_seg.nii.gz',label=True, resize=(155,img_size,img_size))

