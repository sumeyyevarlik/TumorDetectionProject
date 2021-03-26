import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.io import imread_collection
import SimpleITK as sitk
import os
import skimage.io as io
import nibabel as nib
import os
import numpy as np
import nibabel as nib
import imageio
import matplotlib
from PIL import Image
from nibabel.viewers import OrthoSlicer3D


path = 'C:/Users/musta/.spyder-py3/bitirme/DataSet_Train/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T2.54515.mha' 
outpath  = 'C:/Users/musta/.spyder-py3/bitirme/DataSet_Train/HGG/brats_2013_pat0001_1/asd.nii'

img = io.imread(path,plugin='simpleitk')
io.imsave(outpath, img,plugin='simpleitk')

def read_niifile(niifilepath): #read niifile file
    img = nib.load(niifilepath) #download niifile file (actually extract the file)
    img_fdata = img.get_fdata() #Get niifile data
    return img_fdata
 
def save_fig(niifilepath,savepath): #Save as picture
    fdata = read_niifile(niifilepath) #Call the above function to get data
    (x,y,z) = fdata.shape #Get data shape information: (length, width, dimension-number of slices, fourth dimension)
    for k in range(z):
        silce = fdata[:,:,k] #Three positions represent three slices with different angles
    imageio.imwrite(os.path.join(savepath,'{}.png'.format(k)),silce)
                                                                                 #Save the slice information as png format
 
if __name__=='__main__':
    niifilepath='C:/Users/musta/.spyder-py3/bitirme/DataSet_Train/HGG/brats_2013_pat0001_1/asd.nii'
    savepath='C:/Users/musta/.spyder-py3/bitirme/DataSet_Train/HGG/brats_2013_pat0001_1/AD'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    save_fig(niifilepath,savepath)
