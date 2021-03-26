import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.io import imread_collection

import os

DATADIR ="brain_tumor_dataset/"
CATEGORIES = ["yes","no"]

# TODO: RESIM DONDURME OLMALI
# TODO: PIKSEL BOZMA ISLEMLERI OLMALI

#TODO: sadece yeslerle egitilebilir test_train yapinca
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

# PIKSEL BOZMA
# training_data = []
# IMG_SIZE = 50
# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap = 'gray')
# plt.show()



def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

