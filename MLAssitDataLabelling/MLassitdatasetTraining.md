# Build a object detection model

## Use the output from Azure ML Assist data labelling tool and build model

## Prerequisite

- Create a blob container
- Create train folder
- load all the images to use

## update azure ml sdk

```
pip install --upgrade azureml-sdk
```

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

- display azure ml

```
import azureml.core
print(azureml.core.VERSION)
```

- load the workspace

```
# Load workspace
ws = Workspace.from_config()
```

- now display workspace

```
# Choose a name for the run history container in the workspace.
experiment_name = 'labeling_Training_96e319f8'
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

- Create compute cluster

```
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

# Choose a name for your cluster.
amlcompute_cluster_name = "gpu-cluster"

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
    # Create the cluster.
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min_node_count is provided, it will use the scale settings for the cluster.
    compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)

    # For a more detailed view of current AmlCompute status, use get_status().
```

- set up data

```
from azureml.core.datastore import Datastore

# replace with account key for visionnotebooksdata storage account
account_key = os.getenv("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
ds = Datastore.register_azure_blob_container(ws, datastore_name='outofstockds2', container_name='storeimages', 
                                             account_name='storageaccountname', account_key=account_key,
                                             resource_group='rgname')
```

- configure the conda environment

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

- retrieve the dataset

```
from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory, LabeledDatasetTask
from azureml.core import Dataset

# create training dataset
training_dataset_name = experiment_name + "_training_dataset"
if training_dataset_name in ws.datasets:
    training_dataset = ws.datasets.get(training_dataset_name)
    print('Found the training dataset', training_dataset_name)
else:
    training_dataset = _LabeledDatasetFactory.from_json_lines(
        task=LabeledDatasetTask.OBJECT_DETECTION, path=ds.path('FlickrLogos_47/flickr47_train.jsonl'))
    training_dataset = training_dataset.register(workspace=ws, name=training_dataset_name)

# create validation dataset
validation_dataset_name = experiment_name + "_training_dataset"
if validation_dataset_name in ws.datasets:
    validation_dataset = ws.datasets.get(validation_dataset_name)
    print('Found the validation dataset', validation_dataset_name)
else:
    validation_dataset = _LabeledDatasetFactory.from_json_lines(
        task=LabeledDatasetTask.OBJECT_DETECTION, path=ds.path('FlickrLogos_47/flickr47_test.jsonl'))
    validation_dataset = validation_dataset.register(workspace=ws, name=validation_dataset_name)
    
print("Training dataset name: " + training_dataset.name)
print("Validation dataset name: " + validation_dataset.name)
```

```
training_dataset = ws.datasets.get("labeling_Training_96e319f8_training_dataset")
validation_dataset = ws.datasets.get("labeling_Training_96e319f8_training_dataset")
```

- now configure automl settings

```
automl_settings = {
    "iteration_timeout_minutes": 1000,
    "iterations": 1,
    "primary_metric": 'mean_average_precision',
    "featurization": 'off',
    "enable_dnn": True,
    'seed' : 47,
    'deterministic': True
}

if os.getenv("SCENARIO"):
    automl_settings["scenario"] = os.getenv("SCENARIO")

automl_config = AutoMLConfig(task = 'image-object-detection',
                             debug_log = 'automl_errors_1.log',
                             path = project_folder,
                             run_configuration=conda_run_config,
                             training_data=training_dataset,
                             validation_data=validation_dataset,
                             **automl_settings
                            )
```

- run the model

```
remote_run = experiment.submit(automl_config, show_output = True)
```

- status

```
remote_run
```

- wait for model to run

```
remote_run.wait_for_completion()
```