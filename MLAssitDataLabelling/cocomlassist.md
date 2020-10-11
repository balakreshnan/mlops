# Build Custom Vision based Image Labeling using Azure Machine learning - Data Labeling (MLAssist)

## Image Labeling with bounding box using Coco data set

## Prerequistie

- Azure Subscription
- Create a Azure Machine learning services in East US or East US 2
- We need GPU compute for ML assist data labeling
- Make sure request is created to extend GPU quota through support
- Takes 2 days to get the quota extended
- Create a gpu compute like NC series with 0 to 4 nodes
- Create a azure data lake store gen 2 to store the images
- Move the images from coco download web site https://cocodataset.org/#download
- Use the below tutorial to use Azure Data Factory to move the images to storage

https://github.com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/copycocofiles.md

- the above tutorial will guide use to download the image and unzip to blob for further processing
- i am using 2017 Val image zip since it has 5K images.
- i also tried 2017 train images 118K images took about 1 hour and 7 minutes
- Validation 5K images took 3 minute and 6 seconds 03:06

## Steps to create the labeling project

- Create a new data label
- Create a project name coco2017 and then select object detection

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj1.jpg "mlops deploy")

- Create a new Data set

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj2.jpg "mlops deploy")

- Create a new data store 

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj3.jpg "mlops deploy")

- Select the data store
- Enable incremental refresh

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj4.jpg "mlops deploy")

- Create Label next

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj5.jpg "mlops deploy")

- Type any instruction you want to provide for labellers

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj6.jpg "mlops deploy")

- Select ML Assist
- Only available for rectangulat bounding box project for now.
- With out ML assit manual labelling can be done for single class and multiclass image classification

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj7.jpg "mlops deploy")

- Create project
- Now go to the project
- In the dashboard it should display how many photo's to label
- The system automatically picks close minimum images for various labellers
- if more than 2 or more then minimum each will get 75 images to tag
- Labeler progress will also been shown in the dashbaord

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj8.jpg "mlops deploy")

- To Start labelling click Labeled data
- Then click start labeling

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj9.jpg "mlops deploy")

- Now you can select the label and then Select and drag the box for the object
- Once done then click Submit button to progress to next image
- There is hand icon to skip image if there are no objects
- Keep doing more images
- In my case i had to label 250 images and then automatically the training kick starts for ML
- Once training and validation is completed then you will also see Task Prelabeled text in Labeled Data section
- depending on the modelling the dashboard will show how many was prelabeled or manual label

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj10.jpg "mlops deploy")

- The above images say the training and validation is completed but bounding box was not enough for ML assist 
- So the model is saying to do more 311 manual labels to see if it can trigger another model run
- Once the model run's then if it has enought data points the model will draw bounding boxes.
- You can also see the labelers performance on the left

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cocoprj11.jpg "mlops deploy")

- Given the more number images ML assist might need more images to tag
- The System detects based on it's alogrithmn to determine how many images
- Once it is completed then we can export the model as COCO file export
- Create automl model using designer or Azure Machine learning

Have fun