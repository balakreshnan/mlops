# Fast ai Hello world in Azure Machine learning services

## Hello World using fast ai in Azure ML jupyter lab using GPU compute

## Pre-requistie

- Azure Account
- Create Azure Machine learning services
- Create GPU compute - i choosed Standard NC6 with 4 nodes
- Start the VM
- Click Jupyter lab
- I have tensorflow 2.0, pytorch, torchvision already installed

```
if there is no tensorflow, pytorch, tochvision please install them <br/>
pip install --upgrade pip
pip install tensorflow
pip install pytorch, torchvision
```

## Steps to create Hello world in Fast ai

- First install Fast ai

```
pip install fastai
```

- Lets import the necessary library

```
from fastai.vision import *
```

- Download sample minst data set

```
path = untar_data(URLs.MNIST_SAMPLE)
path
```

## Process image from folder

- Nomalize the data and display a sample image. Image is picked from Folder

```
data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)
img,label = data.train_ds[0]
img
```

- Now download model and run reset 18 with above data set

```
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)
```

- Calculate the accuracy

```
accuracy(*learn.get_preds())
```

## Process image from csv file

- Lets load the data from csv file to process

```
data = ImageDataBunch.from_csv(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)
img,label = data.train_ds[0]
img
```

- Train the model 

```
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)
```

- Will build more samples as next steps.