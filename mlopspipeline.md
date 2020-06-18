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

joblib.dump(rfc, "model.joblib")
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

When asked you might have to log in with device login.

Batch inferecing

```
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

cd = CondaDependencies.create(pip_packages=["scikit-learn", "azureml-defaults"])
env = Environment(name="parallelenv")
env.python.conda_dependencies = cd
env.docker.base_image = DEFAULT_CPU_IMAGE
```

Setup Parallel Run

```
from azureml.contrib.pipeline.steps import ParallelRunConfig

parallel_run_config = ParallelRunConfig(
    environment=env,
    entry_script="train.py",
    source_directory=".",
    output_action="append_row",
    mini_batch_size="20",
    error_threshold=1,
    compute_target=cpu_cluster,
    process_count_per_node=2,
    node_count=1
)
```

Parallel Step

```
from azureml.contrib.pipeline.steps import ParallelRunStep

batch_score_step = ParallelRunStep(
    name="parallel-step-test",
    inputs=[data_complete],
    output=output_dir,
    models=[model],
    arguments=["--model_name", "inception",
               "--labels_name", "label_ds"],
    parallel_run_config=parallel_run_config,
    allow_reuse=False
)
``

```
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline

pipeline = Pipeline(workspace=ws, steps=[batch_score_step])
pipeline_run = Experiment(ws, 'batch_scoring_ms').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)
```

```
import pandas as pd

batch_run = next(pipeline_run.get_children())
batch_output = batch_run.get_output_data("scores")
batch_output.download(local_path="inception_results")

for root, dirs, files in os.walk("inception_results"):
    for file in files:
        if file.endswith("parallel_run_step.txt"):
            result_file = os.path.join(root, file)

df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["Filename", "Prediction"]
print("Prediction has ", df.shape[0], " rows")
df.head(10)
```

```
published_pipeline = pipeline_run.publish_pipeline(
    name="Inception_v3_scoring", description="Batch scoring using Inception v3 model", version="1.0")

published_pipeline
```

