# REST endpoint to score a object detection model

## Use the output from Azure ML Assist data labelling tool and build model and create a REST API to score

## Prerequisite

- Object detection model built
- Azure ML workspace
- Complete the model training - https://github.com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/MLAssitObjDetectionTraining.md
- Get the Experiment Name
- Get the AutoML run id of the model ending with _HD_0

## Write scoring script

- import libraries

```
import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace

print("SDK version:", azureml.core.VERSION)
```

- Get the workspace information

```
subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxx"
resource_group="resourcegroupname"

workspace_name = "workspacename"
experiment_name = "labeling_Training_xxxxxx"
run_id = "AutoML_58782d2f-4999-4651-8a40-xxxxxxxxxxx_HD_0"

model_name = 'object-detection-oos'
```

- Register the model

```
from azureml.core import Run

ws = Workspace.get(name=workspace_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)

experiment = Experiment(ws, experiment_name)
run = Run(experiment, run_id)
model = run.register_model(model_name=model_name, model_path='train_artifacts')
```

- Setup AKS cluster using GPU

```
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException

# Choose a name for your cluster
aks_name = "oosaksgpu"

# Check to see if the cluster already exists
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    # Provision AKS cluster with GPU machine
    prov_config = AksCompute.provisioning_configuration(vm_size="Standard_NC6", 
                                                        location="eastus2")
    #prov_config = AksCompute.provisioning_configuration()
    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws, 
                                      name=aks_name, 
                                      provisioning_configuration=prov_config)

    aks_target.wait_for_completion(show_output=True)

aks_target
```

- Write Scoring script for REST API

```
%%writefile score.py

import os
import tempfile
import torch
import pickle

from azureml.contrib.services.aml_request import rawhttp, AMLRequest
from azureml.contrib.services.aml_response import AMLResponse

from azureml.core import Model
from azureml.core.run import _OfflineRun

from azureml.contrib.automl.dnn.vision.common.utils import _set_logging_parameters
from azureml.contrib.automl.dnn.vision.object_detection.common.constants import ArtifactLiterals, FasterRCNNParameters
from azureml.contrib.automl.dnn.vision.object_detection.writers.score import _score_with_model

from azureml.train.automl import constants

def _load_model_wrapper(torch_model_file, model_wrapper_pkl, **model_settings):
    with open(model_wrapper_pkl, 'rb') as fp:
        model_wrapper = pickle.load(fp)

    cuda_available = torch.cuda.is_available()
    print("CUDA available: {}".format(cuda_available))
    
    map_location = None
    if not cuda_available:
        map_location = torch.device('cpu')
        
    model_weights = torch.load(torch_model_file, map_location=map_location)
    model_wrapper.restore_model(model_weights, **model_settings)

    return model_wrapper


def init():
    global model
    
    # Set up logging
    task_type = constants.Tasks.IMAGE_OBJECT_DETECTION
    _set_logging_parameters(task_type, {})

    model_path = Model.get_model_path('object-detection-oos')
    
    model_settings = {'min_size': FasterRCNNParameters.DEFAULT_MIN_SIZE,
                      'box_score_thresh': FasterRCNNParameters.DEFAULT_BOX_SCORE_THRESH,
                      'box_nms_thresh': FasterRCNNParameters.DEFAULT_BOX_NMS_THRESH,
                      'box_detections_per_img': FasterRCNNParameters.DEFAULT_BOX_DETECTIONS_PER_IMG}

    model = _load_model_wrapper(os.path.join(model_path, ArtifactLiterals.MODEL_FILE_NAME),
                                os.path.join(model_path, ArtifactLiterals.PICKLE_FILE_NAME),
                                **model_settings)


@rawhttp
def run(request):
    print("This is run()")
    print("Request: [{0}]".format(request))
    if request.method == 'GET':
        # For this example, just return the URL for GETs.
        response_body = str.encode(request.full_path)
        return AMLResponse(response_body, 200)
    elif request.method == 'POST':
        request_body = request.get_data()
        
        with tempfile.NamedTemporaryFile() as output_filename_fp, \
                tempfile.NamedTemporaryFile(mode="w") as image_list_file_fp, \
                tempfile.NamedTemporaryFile() as image_file_fp:

            image_file_fp.write(request_body)
            image_file_fp.flush()
            
            image_list_file_fp.write(image_file_fp.name)
            image_list_file_fp.flush()

            root_dir = ""
            _score_with_model(model_wrapper=model,
                              run=_OfflineRun(),
                              target_path=None,
                              output_file=output_filename_fp.name,
                              root_dir=root_dir,
                              image_list_file=image_list_file_fp.name,
                              num_workers=0,
                              ignore_data_errors=True)
            output_filename_fp.flush()
            return AMLResponse(output_filename_fp.read(), 200)
    else:
        return AMLResponse("bad request", 500)
```

- Create Environment

```
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment

# manual created environment (should not be needed)
# environment = Environment("ODScoring")
# environment.python.conda_dependencies.add_pip_package("azureml-contrib-automl-dnn-vision")
# environment.docker.base_image = azureml.core.runconfig.DEFAULT_GPU_IMAGE

environment = run.get_environment()
inference_config = InferenceConfig(entry_script='score.py', environment=environment)
```

- Create AKS deployment config

```
from azureml.core.webservice import AksWebservice

gpu_aks_config = AksWebservice.deploy_configuration(autoscale_enabled=True,                                                    
                                                    cpu_cores=1,
                                                    memory_gb=50,
                                                    enable_app_insights=True)
```

- Deploy the model

```
from azureml.core import Model

# Name of the web service that is deployed
aks_service_name = "scoreodgpu"

# Get the registerd model
model = Model(ws, model_name)

# Deploy the model
aks_service = Model.deploy(ws,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=gpu_aks_config,
                           deployment_target=aks_target,
                           name=aks_service_name,
                           overwrite=True)

aks_service.wait_for_deployment(show_output=True)
```

- Make sure REST api is deployed

```
print(aks_target)
print(aks_service.state)
print(aks_service.get_logs())
```

- Now to test the rest api
- Get the REST URL

```
print(aks_service.scoring_uri)
```

- Send an image and score

```
import requests

# URL for the web service
scoring_uri = aks_service.scoring_uri

# If the service is authenticated, set the key or token
key, _ = aks_service.get_keys()

# Load image data
data = open('2020-02-27T09-41-54.000000Z.jpg', 'rb').read()

# Set the content type
headers = {'Content-Type': 'application/octet-stream'}

# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, data, headers=headers)
print(resp.text)
```