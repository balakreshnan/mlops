# Convert AML dataset into JSON for object detection model

## Directory structure

```
blobContainer
            /train
                  /nostock
            /test
                 /nostock
            /annotation.jsonl
```

## Read Azure ML registered data set from Data labelling/ML assist Data Labelling

- read the data set that was registered

```
from azureml.core import Workspace, Dataset
import azureml.contrib.dataset
import pandas as pd
import json

subscription_id = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
resource_group = 'resourcegroup'
workspace_name = 'mlworkspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='OutoFStockTest1_20200808_150508')
dataset.to_pandas_dataframe()
```

- Convert to pandas dataframe

```
objdf = dataset.to_pandas_dataframe()
```

- Convert image_url to string 

```
objdf['image_url'] = objdf['image_url'].astype(str)
```

- Strip the image name for further processing

```
objdf['imgfile'] = objdf['image_url'].str.extract('AmlDatastore://([^/]*[^/\d])\d*\.jpg', expand=False).str.strip()
```

## Loop Each line and save as json lines

- import libraries

```
import numpy
import pandas as pd
```

- now time to loop through each row and split the images and then create json lines

```
x = 1
y = 1

dataall = []
data = {}
data['image_url'] = ""
data['image_details'] = {}
data['label'] = []
with open('annotation.jsonl', 'w') as outfile:    
    for index, row in objdf[['imgfile', 'label']].iterrows():
        filename = row['imgfile'] + ".jpg"
        data = {}
        data['image_url'] = ""
        data['image_details'] = {}
        data['label'] = []
        data['image_url'] = "AmlDatastore://storeimagesjson/train/nostock/" + row['imgfile'] + ".jpg"
        data['image_details']= {
            'format': 'jpg',
            'width': '1024px',
            'height': '740px'
        }
        #print(row['imgfile'], row['label'])
        #data['label'] = row['label']
        for line in row['label']:
            name = line['label']
            topX = str(numpy.float64(line['topX']))
            topY = str(numpy.float64(line['topY']))
            bottomX = str(numpy.float64(line['bottomX']))
            bottomY = str(numpy.float64(line['bottomY']))
            data['label'].append({ 'label': name, 'topX': topX, 'topY': topY, 'bottomX': bottomX, 'bottomY': bottomY, 'isCrowd': 'false', 'mask': {} })
            y = y + 1
        #dataall.append(data)
        json.dump(data, outfile)
        outfile.write('\n')
        x = x + 1
```

- the above should create JSON lines and create annotation.jsonl file.
- download the file and upload to blob storage for training a new model