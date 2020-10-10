# Copy Coco files from web URL into azure blob storage

## Copy Coco sample file to do modelling

## Use Case

To build object detection model we need some data sets. So i decided to use coco data set which is open source.
So our goal is to use Azure Data Factory to copy the zip and deflate the images to a directory to be used in Data labelling project.

## Prerequistie

- Azure subcription
- Create a Azure Data factory resource
- Use copy wizard to move to Blob storage
- Create Azure Blob Storage account or ADLS Gen2

## Steps

- First create copy wizrd

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cococp1.jpg "mlops deploy")

- For the source we are going to copy files from http://images.cocodataset.org/zips/val2017.zip
- Use Http source in Azure Data Factory
- for URL type: http://images.cocodataset.org/zips/val2017.zip
- For Authentication use anonymous
- For the compression select zipflate

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cococp2.jpg "mlops deploy")

- Now configure the destination as Azure data lake store Gen 2
- Configure the folder to save

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/cococp3.jpg "mlops deploy")

- Now Save and publish the pipeline
- Click Trigger now
- Go to Monitor and wait until it completes
- Should copy 5000 images