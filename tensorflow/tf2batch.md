# Tensorflow 2.0 Batch Scoring using Azure Machine learning service ParallelRunStep

## Use Case

For this article i am using Minst data set available to build a devops training script to be used in Azure devops build process

## Requirements

- minst data set
- Azure Devops acoount
- Azure Machine learning services
- Github to store the code
- Download images for scoring from kaggle web site. https://www.kaggle.com/scolianni/mnistasjpg

## Code requirements

- Training code for azure machine learning pipeline
- Tensorflow Train Code

```
Note: use this code as sample or reference because i am using all public available data set and can be changed or moved
```

## Process

First write the code for Batch scoring using tensorflow 2.0.0

pything file name: batchscore3.py. Batchscore file jsut takes the image url passed and open the jpg and then scores and appends the results to txt file. 

What we have to process the output is up to the readers to implement their own logic.

```
# +
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3.6 - AzureML
#     language: python
#     name: python3-azureml
# ---

# # +
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from azureml.core import Model
from azureml.core.model import Model
from azureml.core import Run
from azureml.core.dataset import Dataset

from azureml.core import Workspace, Dataset

from azureml.core.authentication import ServicePrincipalAuthentication


def init():
    global imported_model

    svc_pr_password = "1fY58u0dpP1Yg-i.A~rUp_iz04RxWUFSwv"
 
    svc_pr = ServicePrincipalAuthentication(
        tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47",
        service_principal_id="8a3ddafe-6dd6-48af-867e-d745232a1833",
        service_principal_password="1fY58u0dpP1Yg-i.A~rUp_iz04RxWUFSwv")

    ws = Workspace(
        subscription_id="c46a9435-c957-4e6c-a0f4-b9a597984773",
        resource_group="mlops",
        workspace_name="gputraining",
        auth=svc_pr
        )
    model_root = os.getenv('AZUREML_MODEL_DIR')
    # Pull down the model from the workspace
    model_path = Model.get_model_path("tf-dnn-mnist",6,ws)
    tf_model_folder = 'model'
    # Create a model folder in the current directory
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./outputs/model', exist_ok=True)

    # Construct a graph to execute
    #tf.reset_default_graph()
    #saver = tf.train.import_meta_graph(os.path.join(model_path, 'tf-dnn-mnist.meta'))
    #g_tf_sess = tf.Session()
    #saver.restore(g_tf_sess, os.path.join(model_path, tf_model_folder, 'tf-dnn-mnist.model'))
    #saver.restore(g_tf_sess, os.path.join(model_path, 'tf-dnn-mnist'))
    
    imported_model = tf.saved_model.load(model_path)
    


def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []
    #in_tensor = g_tf_sess.graph.get_tensor_by_name("network/X:0")
    #output = g_tf_sess.graph.get_tensor_by_name("network/output/MatMul:0")
    #in_tensor = g_tf_sess.graph.get_tensor_by_name("Reshape_1:0")
    #output = g_tf_sess.graph.get_tensor_by_name("Reshape_1:0")
    

    for image in mini_batch:
        # Prepare each image
        data = Image.open(image)
        #dataresize = data.resize((28, 28))
        #dataresize = data.thumbnail((28, 28))        
        np_im = np.array(data/np.float32(255.0)).reshape((1, 784))
        
        # Perform inference
        #inference_result = output.eval(feed_dict={in_tensor: np_im}, session=g_tf_sess)        
        out = imported_model(np_im)
        # Find the best probability, and add it to the result list
        best_result = np.argmax(out)
        resultList.append("{}: {}".format(os.path.basename(image), best_result))

    return resultList
```

Lets write the batch score ParallelRunStep pipeline code for azure machine learning services

file name: tf2batch.py

```
import numpy as np
import os
#import matplotlib.pyplot as plt

import azureml
from azureml.core import Workspace

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

from azureml.telemetry import set_diagnostics_collection

set_diagnostics_collection(send_diagnostics=True)

from azureml.core import Workspace, Dataset

from azureml.core.authentication import ServicePrincipalAuthentication
 
svc_pr_password = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
 
svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_password="xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
 
ws = Workspace(
    subscription_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    resource_group="mlops",
    workspace_name="gputraining",
    auth=svc_pr
    )
ws

from azureml.core.dataset import Dataset
web_paths = ['https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
             'https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
             'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
             'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz'
            ]
dataset = Dataset.File.from_files(path = web_paths)

dataset_registered = False
try:
    temp = Dataset.get_by_name(workspace = ws, name = 'bbminsttest')
    dataset_registered = True
except:
    print("The dataset mnist-dataset is not registered in workspace yet.")

if not dataset_registered:
    dataset = dataset.register(workspace = ws,
                               name = 'bbminsttest',
                               description='training and test dataset',
                               create_new_version=True)
# list the files referenced by dataset
dataset.to_path()
dataset = temp

dataset.to_path()

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
cluster_name = "gpucluster1"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())

def_data_store = ws.get_default_datastore()

from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.core import PipelineParameter

pipeline_param = PipelineParameter(name="mnist_param", default_value=dataset)
input_mnist_ds_consumption = DatasetConsumptionConfig("minist_param_config", pipeline_param).as_mount()

from azureml.pipeline.core import Pipeline, PipelineData

output_dir = PipelineData(name="inferences", 
                          datastore=def_data_store, 
                          output_path_on_compute="mnist/results")

from azureml.core import Model
from azureml.core.model import Model
from azureml.core import Run
from azureml.core.dataset import Dataset

from azureml.core import Workspace, Dataset

from azureml.core.authentication import ServicePrincipalAuthentication

model_path = Model.get_model_path("tf-dnn-mnist",6,ws)

model_path

import tensorflow as tf
imported_model = tf.saved_model.load(model_path)

imported_model

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_GPU_IMAGE

batch_conda_deps = CondaDependencies.create(pip_packages=["tensorflow==2.0.0", "pillow",
                                                          "azureml-core", "azureml-dataprep[pandas, fuse]","azureml-sdk[notebooks]"])

batch_env = Environment(name="batch_environment")
batch_env.python.conda_dependencies = batch_conda_deps
batch_env.docker.enabled = True
batch_env.docker.base_image = DEFAULT_GPU_IMAGE

from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.steps import ParallelRunConfig

parallel_run_config = ParallelRunConfig(
    source_directory='',
    entry_script="batchscore3.py",
    mini_batch_size=PipelineParameter(name="batch_size_param", default_value="5"),
    error_threshold=10,
    output_action="append_row",
    append_row_file_name="mnist_outputs.txt",
    environment=batch_env,
    compute_target=compute_target,
    process_count_per_node=PipelineParameter(name="process_count_param", default_value=2),
    node_count=2)

from azureml.pipeline.steps import ParallelRunStep
from datetime import datetime

parallel_step_name = "batchscoring-" + datetime.now().strftime("%Y%m%d%H%M")

from azureml.pipeline.steps import ParallelRunStep

parallelrun_step = ParallelRunStep(
    name=parallel_step_name,
    parallel_run_config=parallel_run_config,
    inputs=[input_mnist_ds_consumption],
    output=output_dir,
    allow_reuse=True
)

from azureml.pipeline.core import Pipeline
from azureml.core.experiment import Experiment

pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
experiment = Experiment(ws, 'batch_scoring')
pipeline_run = experiment.submit(pipeline)

#from azureml.widgets import RunDetails
#RunDetails(pipeline_run).show()

pipeline_run.wait_for_completion(show_output=True)
```