# Azure Machine Learning ML Assisted data labelling dataset to XML for object detection modelling

## Convert the Azure machine learning dataset from ML assist/data labelling output to XML for Object detection modelling

One of the challeng with Azure ML data set is ability to export to format which can be leveraged into existing Mask R-CNN or Fast R-CNN or resnet to train for object detection.

Azure Machine learning ml assist data set is in different format.

This tutorial walks through step take the AML data set anc convert to XML to be able to use in any object detection model

## Pre Requistie

- Azure machine learning service
- Create data labelling project
- enable ML assit 
- label images
- Export label outout as Azure ML dataset
- compute instance

When saved, the dataset section will have the new dataset

Now lets convert to Xml

- Go to compute instance and click jupyter lab/jupyter
- create folder called xml
- create a new notebook with azure ML sdk
- Get workspace details like subscription id, resource group name and ML workspace name

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
# azureml-contrib-dataset of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset
import azureml.contrib.dataset
import pandas as pd

subscription_id = 'replacexxxxxxxxxxxxxxxxxxxxxxxx'
resource_group = 'replaceresourcegroupname'
workspace_name = 'replaceworkspacename'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='OutoFStockTest1_20200808_150508')
dataset.to_pandas_dataframe().head()
```

- Will display only first 5 rows to see the format of AMLdataset produced
- Convert the dataset to dataframe

```
objdf = dataset.to_pandas_dataframe()
```

- Lets convert image_url column to string

```
#convert column to string
objdf['image_url'] = objdf['image_url'].astype(str)
```

- Create a new column called imgfile and pull out the image file name. (here it is jpg file)

```
objdf['imgfile'] = objdf['image_url'].str.extract('AmlDatastore://([^/]*[^/\d])\d*\.jpg', expand=False).str.strip()
```

- Now time to loop all images and create XML file for each image

```
from xml.dom import minidom
import xml.etree.ElementTree as ET
```

- Setup output foler

```
outputfolder = "./xml"
```

- Import necessary libraries

```
import os
import numpy
```

- Time to loop each row and get the image file and labels with cordinates.

```
for index, row in objdf[['imgfile', 'label']].iterrows():
    # print(str(row['imgfile']), row['label'])
    xmlfile = os.path.join(outputfolder, row['imgfile'])
    # create the file structure
    data = ET.Element('annotation')
    folder = ET.SubElement(data, 'folder')
    filename = ET.SubElement(data, 'filename')
    size = ET.SubElement(data, 'size')
    segmented = ET.SubElement(data, 'segmented')    
    filename.text = row['imgfile'] + ".jpg"
    
    for line in row['label']:
        # print(line['label'], line['topX'], line['topY'], line['bottomX'], line['bottomY'])
        object = ET.SubElement(data, 'object')
        name = ET.SubElement(object, 'name')
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        name.text = line['label']
        xmin.text = str(numpy.float64(line['topX']))
        ymin.text = str(numpy.float64(line['topY']))
        xmax.text = str(numpy.float64(line['bottomX']))
        ymax.text = str(numpy.float64(line['bottomY']))
        
    difficult = ET.SubElement(object, 'difficult')
    
    # create a new XML file with the results
    fname = row['imgfile'] + ".xml"
    fullfname = os.path.join(outputfolder,fname)
    #print(fullfname)
    mydata = ET.tostring(data)
    #print(mydata)
    myfile = open(fullfname, "wb")
    myfile.write(mydata)
```

- Above code first expands each row from the data frame. Then get the image name and Labels list
- Loop the label list to get the label and bounding box cordinates
- Format the XML 
- Finally write the xml to disk.