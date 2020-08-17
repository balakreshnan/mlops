# Fast ai classfication model using Azure Machine learning services - ML data labelling output

## Build a classification model using fast ai and azure machine learning services

We are going to build a classfication model from Azure ML data labelling output.

## Pre-requisite

- Azure Account
- Create Azure Machine learning services
- Create GPU compute - I choose Standard NC6 with 4 nodes
- Start the VM
- Click Jupyter lab
- I have tensorflow 2.0, pytorch, torchvision already 

```
if there is no tensorflow, pytorch, tochvision please install them <br/>
pip install --upgrade pip
pip install tensorflow
pip install pytorch, torchvision
```

- Create ML assist data labelling project
- Complete the data labelling process
- Export the data set Azure ML data set
- Get the name of the datasets (both labelled one and also actual images dataset)

## import libraries

```
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
```

## Download the images

```
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
resource_group = 'rgname'
workspace_name = 'mlworkspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='imagedataset')
dataset.download(target_path='.', overwrite=True)
```

## Read the data labelled azure ml data sets

Now lets read the labelled output

```
from azureml.core import Workspace, Dataset
import azureml.contrib.dataset
import pandas as pd

subscription_id = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
resource_group = 'rgname'
workspace_name = 'mlworkspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='mllabelledds_20200808_150508')
dataset.to_pandas_dataframe()
```

Write to proper data set

```
objdf = dataset.to_pandas_dataframe()
```

Now we are going to take the image name from AML data set

```
#convert column to string
objdf['image_url'] = objdf['image_url'].astype(str)

objdf['imgfile'] = objdf['image_url'].str.extract('AmlDatastore://([^/]*[^/\d])\d*\.jpg', expand=False).str.strip()
objdf.head()
objdf[['imgfile', 'label']].head()
```

## Write the images into proper format

```
path/
    /train
          /classes (labels)
```

Create Train, Valid and test folders

Mine looked like

```
path/
    train/
         /nostock

path/
    valid/
         /nostock

path/
    test/
         /nostock
```

Based on the number of classes we need to create the images into 3 folders - train, valid, test

```
import shutil, os
directory = "./"
from pathlib import Path

train_folder = 'train'

for filename1 in os.listdir(directory):
    if filename1.endswith(".jpg"):
        fname = filename.replace(".jpg","")
        dfimg = objdf.loc[objdf['imgfile'] == fname]
        #print(dfimg['imgfile'])
        for index, row in dfimg[['imgfile', 'label']].iterrows():
            for line in row['label']:
                #print(line['label'], filename1)    
                labelname = line['label']        
                train_folder_label = os.path.join(train_folder, labelname)
                if not os.path.exists(train_folder_label):
                    os.makedirs(train_folder_label)
                shutil.copy(filename1, train_folder_label)
                print(train_folder_label, filename1)

train_folder = 'valid'

for filename1 in os.listdir(directory):
    if filename1.endswith(".jpg"):
        fname = filename.replace(".jpg","")
        dfimg = objdf.loc[objdf['imgfile'] == fname]
        #print(dfimg['imgfile'])
        for index, row in dfimg[['imgfile', 'label']].iterrows():
            for line in row['label']:
                #print(line['label'], filename1)    
                labelname = line['label']        
                train_folder_label = os.path.join(train_folder, labelname)
                if not os.path.exists(train_folder_label):
                    os.makedirs(train_folder_label)
                shutil.copy(filename1, train_folder_label)
                print(train_folder_label, filename1)

train_folder = 'test'

for filename1 in os.listdir(directory):
    if filename1.endswith(".jpg"):
        fname = filename.replace(".jpg","")
        dfimg = objdf.loc[objdf['imgfile'] == fname]
        #print(dfimg['imgfile'])
        for index, row in dfimg[['imgfile', 'label']].iterrows():
            for line in row['label']:
                #print(line['label'], filename1)    
                labelname = line['label']        
                train_folder_label = os.path.join(train_folder, labelname)
                if not os.path.exists(train_folder_label):
                    os.makedirs(train_folder_label)
                shutil.copy(filename1, train_folder_label)
                print(train_folder_label, filename1)
```

## Modelling

import warning libraries

```
import warnings
warnings.filterwarnings("ignore")
```

Create the transform function

```
tfms = get_transforms()
```

Create a data set for Fast ai modelling

```
data = ImageDataBunch.from_folder(Path('./'), ds_tfms=tfms, size=24)
```

To diplay the labels or classes 

```
data.classes
```

Now we are configur the model (learner) function. We are using resnet34 (can change to resnet18 or resnet50)

```
learner = cnn_learner(data, models.resnet34, metrics = [accuracy,error_rate], loss_func=torch.nn.CrossEntropyLoss(weight=w))
```

Run Training

```
learner.fit_one_cycle(1,1e-3)
```

Wait for model to complete.

Let's discuss the how to predict

```
img = learner.data.train_ds[0][0]
learner.predict(img)
```

the End