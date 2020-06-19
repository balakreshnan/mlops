# Machine Learning Operations - Data science life cycle process

## Build pipeline to run Training cluster

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsflow1.jpg "mlops deploy")

## Code to run

Let's import necessary packages

```
import os
import urllib
import shutil
import azureml

from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
```

Load workspace config 

```
ws = Workspace.from_config()
```

Configure project name

```
project_folder = './my-project'
os.makedirs(project_folder, exist_ok=True)

experiment = Experiment(workspace=ws, name='my-model')
```

```
output_folder = './outputs'
os.makedirs(output_folder, exist_ok=True)
```

```
result_folder = './results'
os.makedirs(result_folder, exist_ok=True)
```

Load data set from data set

```
from azureml.core import Workspace, Dataset

dataset = Dataset.get_by_name(ws, name='mydataset')
dataset.to_pandas_dataframe()
data_complete = dataset.to_pandas_dataframe()
```

Create compute cluster for training

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpucluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D14_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
```

Config image options

```
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

# Create a new runconfig object
run_amlcompute = RunConfiguration()

# Use the cpu_cluster you created above. 
run_amlcompute.target = cpu_cluster

# Enable Docker
run_amlcompute.environment.docker.enabled = True

# Set Docker base image to the default CPU-based image
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_amlcompute.environment.python.user_managed_dependencies = False

# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])
```

Create a train.py file that train's the model. in our case we choose Random forest

```
%%writefile $project_folder/train.py

import joblib
import os
import urllib
import shutil
import azureml

from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

from sklearn.ensemble import RandomForestClassifier


from azureml.core import Workspace, Dataset

subscription_id = 'subit'
resource_group = 'rg-prj-mlearn'
workspace_name = 'ml-prj-wkspc-001'

ws = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(ws, name='mydataset')
dataset.to_pandas_dataframe()
data_complete = dataset.to_pandas_dataframe()

X = data_complete[[col for col in data_complete.columns if "encoding" in col]]
y = data_complete['touring_flag']

X = X[[col for col in X.columns if col not in ["xxx_range_encoding", "xxx_encoding"]]]

rfc = RandomForestClassifier(n_estimators=20,
                             criterion="entropy",
                             class_weight="balanced",
                             random_state=1,
                             n_jobs=-1,
                             verbose=2)

rfc.fit(X, y)

#joblib.dump(rfc, "model.joblib")
os.makedirs('./outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=rfc, filename='./outputs/sklearn_test_model.pkl')
```

Now time to set the pipeline parameters.

```
from azureml.train.sklearn import SKLearn

# script_params = {
#     '--kernel': 'linear',
#     '--penalty': 1.0,
# }

estimator = SKLearn(source_directory=project_folder, 
#                     script_params=script_params,
                    compute_target=cpu_cluster,
                    entry_script='train.py',
                    pip_packages=['joblib']
                   )
```

Now run the experiment. The experiment can take time and so please submit and wait.

```
run = experiment.submit(estimator)
run.wait_for_completion(show_output=True)
```

```
print(run.get_metrics())
```

```
from azureml.widgets import RunDetails
RunDetails(run).show()
```

```
print(run.get_portal_url())
```

```
print(run.get_file_names())
```

Register the model

```
# register model
model = run.register_model(model_name='sklearn_test',
                           model_path='outputs/sklearn_test_model.pkl')
print(model.name, model.id, model.version, sep='\t')
```

When asked you might have to log in with device login.

Batch inferecing

Create batch_scoring file

```
%%writefile batch_scoring.py
import io
import pickle
import argparse
import numpy as np
import pandas as pd

import joblib
import os
import urllib
import shutil
import azureml

from azureml.core.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def init():
    global touring_model

    model_path = Model.get_model_path("sklearn_test")

    #model_path = Model.get_model_path(args.model_name)
    with open(model_path, 'rb') as model_file:
        touring_model = joblib.load(model_file)


def run(mini_batch):
    # make inference    
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []
    for file in mini_batch:
        input_data = pd.read_parquet(file, engine='pyarrow')
        num_rows, num_cols = input_data.shape
        pred = touring_model.predict(input_data).reshape((num_rows, 1))

    # cleanup output
    result = input_data.drop(input_data.columns[4:], axis=1)
    result['variety'] = pred

    return result
```

Configure input 

```
from azureml.core.datastore import Datastore

batchscore_blob = Datastore.register_azure_blob_container(ws, 
                      datastore_name="scoredataset", 
                      container_name="containername", 
                      account_name="storageaccoutnname",
                      account_key="xxxxxxxxxxxxxxxxxxxxxxx",
                      overwrite=True)

def_data_store = ws.get_default_datastore()
```

Configure output

```
from azureml.core.datastore import Datastore

batchscore_blob_out = Datastore.register_azure_blob_container(ws, 
                      datastore_name="output", 
                      container_name="containername", 
                      account_name="storageaccountname", 
                      account_key="xxxxxxxxxxxxxxxxxxxxxxxxxx",
                      overwrite=True)

def_data_store_out = ws.get_default_datastore()
```

```
from azureml.core.dataset import Dataset
from azureml.pipeline.core import PipelineData

input_ds = Dataset.File.from_files((batchscore_blob, "/"))
output_dir = PipelineData(name="scores", 
                          datastore=def_data_store_out, 
                          output_path_on_compute="results")
```

Setup  environment

```
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

cd = CondaDependencies.create(pip_packages=["scikit-learn", "azureml-defaults"])
env = Environment(name="parallelenv")
env.python.conda_dependencies = cd
env.docker.base_image = DEFAULT_CPU_IMAGE
```

Setup File dataset settings

```
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.steps import ParallelRunConfig

parallel_run_config = ParallelRunConfig(
    #source_directory=scripts_folder,
    entry_script="batch_scoring.py",
    mini_batch_size='1',
    error_threshold=2,
    output_action='append_row',
    append_row_file_name="test_outputs.txt",
    environment=env,
    compute_target=cpu_cluster, 
    node_count=3,
    run_invocation_timeout=600)
```

```
from azureml.pipeline.steps import ParallelRunStep

batch_score_step = ParallelRunStep(
    name="parallel-step-test",
    inputs=[input_ds.as_named_input("input_ds")],
    output=output_dir,
    #models=[model],
    parallel_run_config=parallel_run_config,
    #arguments=['--model_name', 'sklearn_test'],
    allow_reuse=True
)
```

now run the batch inference pipeline

```
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline

pipeline = Pipeline(workspace=ws, steps=[batch_score_step])
pipeline_run = Experiment(ws, 'batch_scoring').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)
```

Monitor

```
from azureml.widgets import RunDetails
RunDetails(pipeline_run).show()
```