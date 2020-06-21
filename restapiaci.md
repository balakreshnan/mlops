# Build Rest API and deploy to ACI as service

Code sample below

```
import os
import urllib
import shutil
import azureml
import joblib

from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
```

```
ws = Workspace.from_config()
```

```
project_folder = './touring-project'
os.makedirs(project_folder, exist_ok=True)

experiment = Experiment(workspace=ws, name='touring-model')
```

```
output_folder = './outputs'
os.makedirs(output_folder, exist_ok=True)
```

```
%%writefile scoring.py
import json
import numpy as np
import os
import pickle
import joblib
import azureml

from azureml.core.model import Model
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def init():
    global touring_model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sklearn_mnist_model.pkl')
    #model_path = './outputs/model.joblib'
    logging.basicConfig(level=logging.DEBUG)
    print(Model.get_model_path(model_name='touring-model'))
    model_path = Model.get_model_path(model_name = "touring-model")
    with open(model_path, 'rb') as model_file:
        touring_model = joblib.load(model_file)
    #model = joblib.load(model_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        # make prediction
        X = data[[col for col in input_data.columns if "encoding" in col]]
        y = data['touring_flag']

        X = X[[col for col in X.columns if col not in ["income_premium_range_encoding", "networth_encoding"]]]
        y_hat = touring_model.predict(X)
        # you can return any data type as long as it is JSON-serializable
        return y_hat.tolist()
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})
```

```
from azureml.core import Environment 
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.runconfig import DEFAULT_CPU_IMAGE 
# cd = CondaDependencies.create(pip_packages=["sklearn", "azureml-defaults"]) 
env = Environment(name="parallelenv") 
# env.python.conda_dependencies = cd 
env.docker.base_image = DEFAULT_CPU_IMAGE
```

```
# from azureml.core.environment import Environment
# from azureml.core.conda_dependencies import CondaDependencies

# myenv = Environment(name="myenv")
conda_dep = CondaDependencies()

# Installs numpy version 1.17.0 conda package
conda_dep.add_conda_package("numpy==1.17.0")

# installs joblib
conda_dep.add_conda_package("joblib")

# install azure
conda_dep.add_pip_package("azureml-defaults")

# Installs pillow package
conda_dep.add_pip_package("pillow")
conda_dep.add_pip_package("scikit-learn")

# Adds dependencies to PythonSection of myenv
env.python.conda_dependencies=conda_dep
```

```
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig

# myenv = Environment.get(workspace=ws, name='myenv', version='1')
inference_config = InferenceConfig(entry_script='scoring.py', environment=env)
```

```
import json
import numpy as np
import os
import pickle
import joblib
import azureml

from azureml.core.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

global model

#model_path = Model.get_model_path("touring-model")
#print(model_path)
#model_path = "outputs/touringmodel.pkl"
#model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
#model = joblib.load(model_path)
model = Model(ws, 'touring-model')
#model_path = Model.get_model_path(model_name = "touring-model")
```

```
print(model.url)
```

```
print('Name:', model.name)
print('Version:', model.version)
```

```
print(model.get_sas_urls())
```

```
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 3, memory_gb = 15, location = "centralus")
service = Model.deploy(ws, "aciservice", [model], inference_config, deployment_config)
service.wait_for_deployment(show_output = True)
print(service.state)
```

```
print(service.state)
```

```
print(service.scoring_uri)
print(service.swagger_uri)
```

```
import requests 
import json 

# send a random row from the test set to score 
# random_index = np.random.randint(0, len(X_test)-1) 
input_data = json.dumps({
    "input_data": {
        "data": [
            [
                100001490.0,
                9767047.0,
                42.334468,
                -88.183048,
                401,
                20050307.0,
                "NaN",
                1,
                0,
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                "SPORTSTER 883",
                20050307.0,
                44.0,
                0.0,
                12.0,
                13.0,
                "F",
                1.0,
                "S",
                1,
                0,
                76.0,
                "R",
                1,
                1,
                0,
                1,
                0.466489,
                0.475747,
                0.508649,
                0.490662,
                0.497505,
                0.498212,
                0.448070
            ]
        ]
    }
}
)

headers = {"Content-Type":"application/json"} 

resp = requests.post(service.scoring_uri, input_data, headers=headers)
print(resp)
```

```
service.delete()
```