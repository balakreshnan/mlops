# View output from batch score file

```
import os
import urllib
import shutil
import azureml
import pandas as pd
import pyarrow

from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
```

```
ws = Workspace.from_config()
```

```
#!pip install azure-storage-blob --upgrade
```

```
from azure.storage.blob import BlockBlobService
#from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pandas as pd
import tables
import time

STORAGEACCOUNTNAME= "mlstorageaccount"
STORAGEACCOUNTKEY= "key"
LOCALFILENAME= "xxxx_outputs.txt"
CONTAINERNAME= "azureml-blobstore-0fb30f4c-f96f-4ccb-98e8-b741ce9e8f19"
BLOBNAME= "azureml/aed286d6-7bea-433d-aa45-0161d231dee8/scores/tuoring_outputs.txt"
#BLOBNAME= "tuoring_outputs.txt"
```

```
#download from blob
t1=time.time()
blob_service=BlockBlobService(account_name=STORAGEACCOUNTNAME,account_key=STORAGEACCOUNTKEY)
blobname=blob_service.get_blob_to_path(CONTAINERNAME,BLOBNAME,LOCALFILENAME)
#blob_service.get_blob_to_path(CONTAINERNAME,BLOBNAME,LOCALFILENAME)
t2=time.time()
print(("It takes %s seconds to download "+str(blobname)) % (t2 - t1))
```

```
dataframe_blobdata = pd.read_csv(LOCALFILENAME)
```

```
dataframe_blobdata.head(5)
```

End