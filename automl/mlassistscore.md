# Score ML Assist Data labelling output using azure Machine learning SDK

## Use Case

Azure Machine learning services MLAssit data labelling project produces model to draw bounding box. The goal is to use those model to score more images

## Requirements

- Azure Machine learning services
- Create a blob and upload the training images
- Create file dataset
- Create a data labeling project enable ml assist
- Tag few images and then wait until the ML assist can create a model
- Once the model is available then collect the experiment name and run id

## Offline Scoring

So this tutorial is for scoring a model once it has completed previously

- First create a new notebook
- a

```
import tempfile
from azureml.core.script_run_config import ScriptRunConfig
```

```
from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.model import Model
```

- Configure the workspace

```
ws = Workspace.from_config()
ws
```

- Now lets load the experiment which we want to do the scoring

```
experiment = Experiment(workspace=ws, name='labeling_Training_xxxxx')
experiment
```

- Get current id

```
currentrun=Run(experiment, "AutoML_e2a27bf7-fde4-4d4e-xxxx-xxxxxxxxx")
currentrun.id
```

- Now time to get the auto ml run id

```
# Load training script run corresponding to AutoML run above.
training_run_id = currentrun.id + "_HD_0"
training_run = Run(experiment, training_run_id)
```

- Setup the data set. Here upload the new images to score in a directory and then setup a dataset in azure ML

```
inference_dataset_name = "oosstockinference"
inference_dataset = ws.datasets.get(inference_dataset_name)
```

```
inference_dataset.id
experiment.name
```

- Setup a GPU compute

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
```

```
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
from azureml.core import ScriptRunConfig, RunConfiguration, Experiment
```

- Now setup environment 

```
# Create a run definition for inference runs
# Get environment from training run
training_env = training_run.get_environment()
training_compute_target = training_run.get_details()["target"]

# create conda run config and set properties from training run
conda_run_config = RunConfiguration(framework="python")

conda_run_config.target = compute_target
conda_run_config.environment.docker.enabled = True
conda_run_config.environment.docker.base_image = training_env.docker.base_image
conda_run_config.environment.python.conda_dependencies = training_env.python.conda_dependencies
```

- Setup the argument for scoring like experiment name, training id and data set it

```
# Inference script run arguments
arguments = [
        "--run_id", training_run_id,
        "--experiment_name", experiment.name,
        "--input_dataset_id", inference_dataset.id
    ]
```

- Let's setup scoring or inferencing code

```
output_prediction_file = "./outputs/predictions.txt"
scoring_args = arguments + ["--output_file", output_prediction_file]
with tempfile.TemporaryDirectory() as tmpdir:
    # Download required files from training run into temp folder.
    entry_script_name = "score_script.py"
    output_path = os.path.join(tmpdir, entry_script_name)
    training_run.download_file("train_artifacts/" + entry_script_name, os.path.join(tmpdir, entry_script_name))
    
    script_run_config = ScriptRunConfig(source_directory=tmpdir,
                                        script=entry_script_name,
                                        run_config=conda_run_config,
                                        arguments=scoring_args)
    scoring_run = experiment.submit(script_run_config) 
```

```
scoring_run
```

- Wait for it complete

```
scoring_run.wait_for_completion()
```

- wait for score run completed

```
import json

scoring_run.download_file(output_prediction_file, output_file_path=output_prediction_file)
with open(output_prediction_file) as predictions:
    #number_of_lines_in_prediction_file = len(predictions.readlines())
    for line in predictions:
        print(line + '\n')  
```

should get a display of images names and what was produced