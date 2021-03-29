import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt
import numpy
import SimpleITK
import matplotlib.pyplot as plt
 
def sitk_show(img, title=None, margin=0.0, dpi=40):
    nda = SimpleITK.GetArrayFromImage(img)
    # spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    # extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
 
    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)
 
    if title:
        plt.title(title)
 
    plt.show()
 
# Paths to the .mhd files
filenameT1 = "C:/Users/musta/.spyder-py3/bitirme/DataSet_Train/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T1.54513.mha"
filenameT2 = "C:/Users/musta/.spyder-py3/bitirme/DataSet_Train/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T2.54515.mha"

# Slice index to visualize with 'sitk_show'
idxSlice = 1
 
# int label to assign to the segmented gray matter
labelGrayMatter = 1
 
imgT1Original = SimpleITK.ReadImage(filenameT1)
imgT2Original = SimpleITK.ReadImage(filenameT2)
 
while idxSlice < 154:
    idxSlice = idxSlice + 1
    sitk_show(SimpleITK.Tile(imgT1Original[:, :, idxSlice],
                         imgT2Original[:, :, idxSlice]))
