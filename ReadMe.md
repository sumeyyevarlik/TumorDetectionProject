# Tumor Detection and Segmentation

## Abstract
A brain tumor is the growth of abnormal cells that occur in replication during brain cell regeneration. In this project, the U-net deep learning 
architecture, in which an artificial neural network developed for biomedical image segmentation detects the presence of tumor by reading brain MR 
images, enables the classification of tumors as necrosis, edema and contrast-enhancing tumors. As a result, a system that reports the MR image and 
the patient's condition has been developed

## Dataset
BraTS data set was used for this study. The training set contains images from 370 patients, including 294 HGG and 76 LGG. There are four sequences 
for each patient's image in dataset : T1, T1ce, T2, and FLAIR.  
The properties of each of these sequences are different from each other. Tag information for segmentation is available in the dataset. Label 0 
segmentation of the radiologist, label 1 full tumor, label 2 necrosis, label 3 non-edema tumor, label 4 enhancing tumor.

   <img src="https://user-images.githubusercontent.com/50299378/120863728-b0ae2800-c593-11eb-826b-cf4faa8cce07.png" width="545" height="155"><br>
   <sub><b>Figure 1: </b> Different sequences of brain images </sub><br> 

## Data Preprocess
Due to the scarcity of MR data, the number of data was increased with the Data Augmentation method. In this study, two separate operations were performed on flair and segment images to segment brain tumors. In segment visuals, all label values are set to the same value. Since there is no tag value in flair images, zero normalization is applied to increase it. The number of samples was increased by applying rotation, horizontal flip and vertical flip to the images. 70 to 130 slices are used for preprocessing. These slices were chosen because the training data is unlikely to have any tumors in the remaining slices of the image. Slices are then normalized to zero mean using the mean and standard deviation. The reproduced data was transformed into an array and the necessary data for training was created. 

![image](https://user-images.githubusercontent.com/50299378/120865344-890c8f00-c596-11eb-95a5-b3bb528183f5.png)<br>
 <sub><b>Figure 2: </b> Data augmentation operations </sub><br>
 
While performing tumor segmentation, T1ce images are used to make the parts of the tumor more prominent. When processing these images, the size of the tumor remains very small compared to the overall x-ray size, so the results are not very healthy when making predictions with the model. Therefore, cropping is applied to the part where the tumor is on these images.

![image](https://user-images.githubusercontent.com/50299378/120865706-3c758380-c597-11eb-8e53-8af6ea55fb43.png)<br>
<sub><b>Figure 3: </b> Tumor image that does not fit and fits in the frame after cropping </sub><br>

## Network Architecture
In the study, experiments have been carried out on the model using U-Net network architecture. The aim at this stage is to obtain the optimum result.
The network is based entirely on convolutional neural networks, and its architecture has been extended to work with less educated images and provide more precise segmentation. 
The proposed 9-layer U-net Structure was used for full tumor segmentation. There are four differences between the U-Net architecture used and the original U-Net architecture:
+ Batch normalization layer added after each convolution layer.
+ Used same padding on convolution layers to keep feature map size unchanged.
+ The filter number of the last convolution layer is the same for dual segmentation.
+ Input channels are given in duplicate because T2 and Flair images are used for full tumor segmentation.


## Train
The model was trained by reading the Flair, T2 and Flair+T2 images to be used for the model. The model
The model, which can easily detect full tumors, cannot successfully detect enhancing and necrotic tumor types. This is because they are very small relative to the brain image.

<img src="https://user-images.githubusercontent.com/50299378/120868741-5b771400-c59d-11eb-9356-f5055e0d1272.png" width="450" height="150"><br>
<sub><b>Figure 4: </b> Image of flair, t2 and concatenated forms </sub><br>

For this T1C images by giving to the model will be provided a better learning. Training continues by gives all these pictures to the model as a series. For the training of the second and third models, the data prepared by applying cropping on the T1ce images will be used. The data divided into two as enhancing and non-edema according to the label 
value is thrown into two arrays defined by cropping and applying threshold. A 7-layered U-Net model is applied for each array, and model training is carried out with the necessary parameters. At the end of the training, 2 more models are created, where the model can make predictions for enhancing and non-edema tumor sections. The weight values resulting from the training are saved to be used in the last segmentation process.
The previously trained model can successfully detect tumors on MR images such as Flair, T2 and generate predictions.  The model, which had difficulties in detecting small tumors in T1C MR images and produced erroneous results, has been able to detect small tumors successfully thanks to cropping processes.

<img src="https://user-images.githubusercontent.com/50299378/120868252-4e0d5a00-c59c-11eb-831f-090390a18fb5.png" width="470" height="150"><br>
<sub><b>Figure 5: </b> Cropped view of the tumor area and the non-edematous area </sub><br>

## Result
Flair and t2-sequence images are used in education to obtain the full tumor. At the end of the training, the model successfully detected the full tumor. Afterwards, the t1ce sequenced images in the dataset were used to detect the enlarged and non-edematous tumor as it strengthens the features of the tumor border and makes the border clear and easily distinguishable. In the model created here, the detection of enlarged and non-edema tumor was also successfully achieved. Finally, segmentation was performed by superimposing the obtained images, and by testing the model, images similar to the segmentation data given in the training data could be obtained.

![image](https://user-images.githubusercontent.com/50299378/120869477-400d0880-c59f-11eb-87b9-ebf901a46d7e.png)<br>
<sub><b>Figure 6: </b> With and without augmentation dice coef compare </sub><br>
<br>
![image](https://user-images.githubusercontent.com/50299378/120868001-bdcf1500-c59b-11eb-90df-a2b62e3144be.png)<br>
<sub><b>Figure 7: </b> Segmented image and image of data predicted </sub><br>
<br>
<img src="https://user-images.githubusercontent.com/50299378/120869057-28815000-c59e-11eb-98d3-c4b73273272f.png" width="630" height="180"><br>
<sub><b>Figure 8: </b> All tumor type predictions of the model </sub><br>



