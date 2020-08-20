# Build a object detection model

## Use the output from Azure ML Assist data labelling tool and build model

## Prerequisite

- Create a blob container
- Create train folder
- Move all the images based on the directory from https://github.com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/AMLtoJson.md
- Directory structure should follow "AmlDatastore://storeimagesjson/train/nostock/imagname.jpg"
- for the above example storeimagesjson is the container name
- train is main folder and i created class as another folder called nostock
- then place the images there
- For annotation file place it under train folder it self
- Azure ML service only these regions are suported: eastus, eastus2
- i am testing only eastus2

```
blobContainer
            /train
                  /nostock
            /test
                 /nostock
            /annotation.jsonl
```

- Above is am example how to store the file structure for object detection model training (create new model)

## update azure ml sdk

```
pip install --upgrade azureml-sdk
pip install --upgrade azureml-contrib-automl-dnn-vision
print("SDK version:", azureml.core.VERSION)
```

- minimum version needed: SDK version: 1.12.0

## Build New model training code.

- Import necessary libraries

```
import logging
import os
import csv

import numpy as np
import pandas as pd
from sklearn import datasets

import azureml.core
from azureml.core import Run, Workspace
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
import azureml.dataprep as dprep
from azureml.core.dataset import Dataset
```

- Get the workspace details to create train run.

```
ws = Workspace.from_config()
```

- Setup the environment and project variables

```
experiment_name = 'labeling_Training_xxxxxxx'
project_folder = './project'

experiment = Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace Name'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T
```

- experiment name is the one i pulled from datalabelling project where in the project home page you should see the train run experiment name. Copy and paste that here
- Setup compute 

```
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

# Choose a name for your cluster.
amlcompute_cluster_name = "gpucluster"

found = False
# Check if this compute target already exists in the workspace.
cts = ws.compute_targets
if amlcompute_cluster_name in cts and cts[amlcompute_cluster_name].type == 'AmlCompute':
    found = True
    print('Found existing compute target.')
    compute_target = cts[amlcompute_cluster_name]

if not found:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_NC6",
                                                                max_nodes = 4)

    # Create the cluster.\n",
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min_node_count is provided, it will use the scale settings for the cluster.
    compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)

     # For a more detailed view of current AmlCompute status, use get_status().
```

- Now time to setup the data set read images and annoation files. Note dataset and containername should be same

```
from azureml.core.datastore import Datastore

account_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ds = Datastore.register_azure_blob_container(ws, datastore_name='containername', container_name='containername', 
                                             account_name='storageaccountname', account_key=account_key,
                                             resource_group='resourcegroupname')
```

- Create the build conda dependencies

```
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import pkg_resources

# create a new RunConfig object
conda_run_config = RunConfiguration(framework="python")

# Set compute target to AmlCompute
conda_run_config.target = compute_target
conda_run_config.environment.docker.enabled = True
conda_run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_GPU_IMAGE

cd = CondaDependencies()

conda_run_config.environment.python.conda_dependencies = cd
```

- Load the training images and annotation file

```
from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory
from azureml.core import Dataset

# create training dataset
training_dataset_name = experiment_name + "_training_dataset"
if training_dataset_name in ws.datasets:
    training_dataset = ws.datasets.get(training_dataset_name)
    print('Found the dataset', training_dataset_name)
else:
    training_dataset = _LabeledDatasetFactory.from_json_lines(
        task="ImageClassification", path=ds.path('train/annotation.jsonl'))
    training_dataset = training_dataset.register(workspace=ws, name=training_dataset_name)
```

- Now create the AUTOML experiment

```
automl_settings = {
    "iteration_timeout_minutes": 1000,
    "iterations": 1,
    "primary_metric": 'mean_average_precision',
    "featurization": 'off',
    "enable_dnn": True,
    "dataset_id": training_dataset.id
}
automl_config = AutoMLConfig(task = 'image-object-detection',
                             debug_log = 'automl_errors_1.log',
                             path = project_folder,
                             run_configuration=conda_run_config,
                             training_data = training_dataset,
                             label_column_name = "label",
                             **automl_settings
                            )
```

- Run the experiment

```
remote_run = experiment.submit(automl_config, show_output = False)
```

- wait for experiment to complete

```
remote_run
remote_run.wait_for_completion()
```

- Now the model is ready