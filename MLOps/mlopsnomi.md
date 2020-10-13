# ML ops using Azure Devops and Github

## Use Azure DevOps for deploy models

## Prerequstie

- Create Score file

```
import io
import pickle
import argparse
import numpy as np

from azureml.core.model import Model
from sklearn.linear_model import LogisticRegression

from azureml_user.parallel_run import EntryScript


def init():
    global iris_model

    logger = EntryScript().logger
    logger.info("init() is called.")

    parser = argparse.ArgumentParser(description="Iris model serving")
    parser.add_argument('--model_name', dest="model_name", required=True)
    args, unknown_args = parser.parse_known_args()

    model_path = Model.get_model_path(args.model_name)
    with open(model_path, 'rb') as model_file:
        iris_model = pickle.load(model_file)


def run(input_data):
    logger = EntryScript().logger
    logger.info("run() is called with: {}.".format(input_data))

    # make inference
    input_data = input_data.iloc[:,:-1]
    num_rows, num_cols = input_data.shape
    pred = iris_model.predict(input_data).reshape((num_rows, 1))

    # cleanup output
    result = input_data.drop(input_data.columns[4:], axis=1)
    result['variety'] = pred

    return result
```

- Create Main run file

```
import os
import azureml.core
from azureml.core import Workspace, Dataset, Datastore, ComputeTarget, RunConfiguration, Experiment
from azureml.core.runconfig import CondaDependencies
from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig
from azureml.pipeline.core import Pipeline
# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)
from azureml.core.authentication import ServicePrincipalAuthentication

import azureml.core
print(azureml.core.VERSION)

# load workspace
#workspace = Workspace.from_config()
tenant_id="xxxxxx"
client_id="xxxxxx"
svc_pr_password="xxxxxxx"

subscription_id="xxxxxx"
my_rg_name="rgname"
my_workspace_name="ML workspace"

svc_pr = ServicePrincipalAuthentication( tenant_id=tenant_id, 
                                         service_principal_id=client_id, 
                                         service_principal_password=svc_pr_password)
                                         
workspace = Workspace(  subscription_id=subscription_id, 
                        resource_group=my_rg_name, 
                        workspace_name=my_workspace_name, 
                        auth=svc_pr ) 

print('Workspace name: ' + workspace.name, 
      'Azure region: ' + workspace.location, 
      'Subscription id: ' + workspace.subscription_id, 
      'Resource group: ' + workspace.resource_group, sep='\n')

from azureml.core import Datastore, Dataset

default_store = workspace.get_default_datastore() 
adls2_dstore = Datastore.get(workspace,'irisscoredata')

# iris_ds = Dataset.File.from_files(adls2_dstore.path('iris_data.csv'))
iris_ds = Dataset.Tabular.from_delimited_files(adls2_dstore.path('iris_data.csv'))
# iris_ds = Dataset.Tabular.from_delimited_files(adls2_dstore.path('unprepared_yellow.csv'))

iris_ds = iris_ds.register(workspace, 'iris_data', create_new_version=True)
df = iris_ds.to_pandas_dataframe()
df = df.iloc[:,:-1]
df.head()

from azureml.core.compute import ComputeTarget, AmlCompute

compute_name = "cpu-cluster"
vm_size = "STANDARD_NC6"
if compute_name in workspace.compute_targets:
    compute_target = workspace.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target: ' + compute_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,  # STANDARD_NC6 is GPU-enabled
                                                                min_nodes=0,
                                                                max_nodes=4)
    # create the compute target
    compute_target = ComputeTarget.create(
        workspace, compute_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

    # For a more detailed view of current cluster status, use the 'status' property
    print(compute_target.status.serialize())
    
scripts_folder = "Code"
script_file = "iris_score.py"

# peek at contents
with open(os.path.join(scripts_folder, script_file)) as inference_file:
    print(inference_file.read())
    
from azureml.core.model import Model

# model_datastore.download('iris_model.pkl')

# register downloaded model
# model = Model.register(model_path = "iris_model.pkl/iris_model.pkl",
model = Model.register(model_path = "./iris_model.pkl",
                       model_name = "iris-prs", # this is the name the model is registered as
                       tags = {'pretrained': "iris"},
                       workspace = workspace)

from azureml.data import OutputFileDatasetConfig
# get the default datastore of the workspace
#datastore = workspace.get_default_datastore()

# write output to datastore under folder `outputdataset/parallelrun` and registger it as a dataset after the experiment completes
# prepared_covid_ds = OutputFileDatasetConfig(destination=(adls2_dstore, 'outputdataset/parallelrun')).register_on_complete(name='prepared_covid_ds')

predicted_ds = OutputFileDatasetConfig(destination=(adls2_dstore, 'outputdataset/parallelrun')).register_on_complete(name='predicted_ds')

from azureml.core import Environment
from azureml.core.runconfig import CondaDependencies, DEFAULT_CPU_IMAGE

# predict_conda_deps = CondaDependencies.create(pip_packages=["scikit-learn==0.20.3",
#                                                             "azureml-core", "azureml-dataset-runtime[pandas,fuse]"])

# batch_conda_deps = CondaDependencies.create(pip_packages=['pandas','azureml-sdk==0.1.0.*','scikit-learn==0.20.3','azureml-dataset-runtime[pandas,fuse]'],
#                                             pip_indexurl='https://azuremlsdktestpypi.azureedge.net/dev/aml/office/134157926D8F')

batch_conda_deps = CondaDependencies.create(pip_packages=['pandas','azureml-sdk==1.15.0','scikit-learn==0.20.3','azureml-dataset-runtime[pandas,fuse]'])

batch_env = Environment(name="batch_environment")
batch_env.python.conda_dependencies = batch_conda_deps
batch_env.docker.enabled = True
batch_env.docker.base_image = DEFAULT_CPU_IMAGE

from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig

# In a real-world scenario, you'll want to shape your process per node and nodes to fit your problem domain.
parallel_run_config = ParallelRunConfig(
    source_directory=scripts_folder,
    entry_script=script_file,  # the user script to run against each input
    mini_batch_size='1KB',
    # mini_batch_size='10',
    error_threshold=5,
    output_action='append_row',
    append_row_file_name="iris_outputs_2.txt",
    environment=batch_env,
    compute_target=compute_target, 
    node_count=2,
    run_invocation_timeout=600
)

distributed_csv_iris_step = ParallelRunStep(
    name='example-iris',
    parallel_run_config=parallel_run_config,
    inputs=[iris_ds.as_named_input('iris_data')],
    # output=output_folder,
    output=predicted_ds,
    arguments=['--model_name', 'iris-prs'],
    allow_reuse=False
)

from azureml.core import Experiment
from azureml.pipeline.core import Pipeline

pipeline = Pipeline(workspace=workspace, steps=[distributed_csv_iris_step])

pipeline_run = Experiment(workspace, 'iris-prs').submit(pipeline)

pipeline_run.wait_for_completion()


dataset = Dataset.get_by_name(workspace, name='predicted_ds')
dataset.download(target_path='.', overwrite=False)

import pandas as pd

fdf = pd.read_csv("./iris_outputs_2.txt", delimiter=" ")
fdf
```

## Azure DevOps Process

- Create a DevOps Project
- Create Service Connections
- one for Azure ML workspace
- One for ARM Scripts


## Steps

- Create a pipeline
- Select Classic editor

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsmi.jpg "mlops Parallel Step")

- Select the github repo where the code it
- Next Select the agent configuration

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsmi1.jpg "mlops Parallel Step")

- Make sure ubuntu 16.04 is selected
- Leave the other agent configuration as default what is populated
- Add another task for python version and select 3.6

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsmi2.jpg "mlops Parallel Step")

- Now add another task for Bash script and select inline script 

```
pip install --upgrade azureml-sdk
pip install azureml-dataprep[pandas]
```

- Leave others the same

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsmi3.jpg "mlops Parallel Step")

- Now validate if packages are installed - This is only for validation
- Create a bash task and type the command

```
pip list
```

- leave others as default

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsmi4.jpg "mlops Parallel Step")

- now lets add another python script task
- Select file option
- Select the proper python main train model file
- in our case it is ModelTraining.py

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsmi5.jpg "mlops Parallel Step")

- Now save the queue the build pipeline
- wait until it completes

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/mlopsmi6.jpg "mlops Parallel Step")