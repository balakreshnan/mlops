# Train and hyper parameter tuning model using Azure Machine learning

## Use case

Train a basic model using scikit RandomClassifier and validate and get accuracy and then run hyper parameter tuning.<br/>
In this example we will use Azure machine learning pipeline to train which we can use this for Azure DevOps for CI/CD

## Steps

First Train a basic model.

Lets import all the necessary package

```
import os
import urllib
import shutil
import azureml
import pandas as pd

from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment, Workspace, Run, Dataset

import argparse
import os
import pandas as pd
import numpy as np
import pickle
import json

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import azureml.core
from azureml.core import Run
from azureml.core.model import Model
from azureml.core import Workspace, Dataset
from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

```

Set authentication to run the model using service principal

```
import os
from azureml.core.authentication import ServicePrincipalAuthentication
 
svc_pr_password = os.environ.get("AZUREML_PASSWORD")
 
svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_password="xxxxxxxxxxxxxxxxxxxxx")
 
ws = Workspace(
    subscription_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    resource_group="mlops",
    workspace_name="mlopsdev",
    auth=svc_pr
    )
 
print("Found workspace {} at location {}".format(ws.name, ws.location))
```

Now time to setup environment and project folder to use

```
project_folder = './diabetes-project'
os.makedirs(project_folder, exist_ok=True)

experiment = Experiment(workspace=ws, name='diabetes-model')
```

```
output_folder = './outputs'
os.makedirs(output_folder, exist_ok=True)
```

```
result_folder = './results'
os.makedirs(result_folder, exist_ok=True)
```

Now load the data: (this data setup is publicly available one and doesn't have real data)

```
df = pd.read_csv('https://mlopssa.blob.core.windows.net/chd-dataset/framingham.csv')
```

Now time to clean up the data set and get only what we need for feature

```
# create a boolean array of smokers
smoke = (df['currentSmoker']==1)
# Apply mean to NaNs in cigsPerDay but using a set of smokers only
df.loc[smoke,'cigsPerDay'] = df.loc[smoke,'cigsPerDay'].fillna(df.loc[smoke,'cigsPerDay'].mean())

# Fill out missing values
df['BPMeds'].fillna(0, inplace = True)
df['glucose'].fillna(df.glucose.mean(), inplace = True)
df['totChol'].fillna(df.totChol.mean(), inplace = True)
df['education'].fillna(1, inplace = True)
df['BMI'].fillna(df.BMI.mean(), inplace = True)
df['heartRate'].fillna(df.heartRate.mean(), inplace = True)

# Features and label
features = df.iloc[:,:-1]
result = df.iloc[:,-1] # the last column is what we are about to forecast
```

Split the data set

```
# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(features, result, test_size = 0.2, random_state = 14)
```

Train model

```
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)
```

Run feature importance and filter out unwanted columns

```
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
```

```
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.12
sfm = SelectFromModel(clf, threshold=0.12)

# Train the selector
sfm.fit(X_train, y_train)

# Features selected
featureNames = list(features.columns.values) # creating a list with features' names
print("Feature names:")
for featureNameListindex in sfm.get_support(indices=True):
    print(featureNames[featureNameListindex])

# Feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# With only imporant features. Can check X_important_train.shape[1]
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)
#Y_important_test = sfm.transform(y_test)

rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
rfc.fit(X_important_train, y_train)
```

Now predict with remaining data 

```
preds = rfc.predict(X_important_test)
```

Get metric out of above model

```
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, preds))
```

Now time to setup pipeline. So lets configure the compute

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "diabetescluster"

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

Setup compute variables and environment

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

Time to write train.py file

```
%%writefile $project_folder/train.py

import joblib
import os
import urllib
import shutil
import azureml
import argparse
import pandas as pd
import numpy as np
import pickle
import json
from sklearn import metrics

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

from sklearn.ensemble import RandomForestClassifier


from azureml.core import Workspace, Dataset

from azureml.core.authentication import ServicePrincipalAuthentication
 
svc_pr_password = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
 
svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_password="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
 
ws = Workspace(
    subscription_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    resource_group="mlops",
    workspace_name="mlopsdev",
    auth=svc_pr
    )

#dataset = Dataset.get_by_name(ws, name='touringdataset')
#dataset.to_pandas_dataframe()
data_complete = df = pd.read_csv('https://mlopssa.blob.core.windows.net/chd-dataset/framingham.csv')

# Get the experiment run context
run = Run.get_context()

# create a boolean array of smokers
smoke = (df['currentSmoker']==1)
# Apply mean to NaNs in cigsPerDay but using a set of smokers only
df.loc[smoke,'cigsPerDay'] = df.loc[smoke,'cigsPerDay'].fillna(df.loc[smoke,'cigsPerDay'].mean())

# Fill out missing values
df['BPMeds'].fillna(0, inplace = True)
df['glucose'].fillna(df.glucose.mean(), inplace = True)
df['totChol'].fillna(df.totChol.mean(), inplace = True)
df['education'].fillna(1, inplace = True)
df['BMI'].fillna(df.BMI.mean(), inplace = True)
df['heartRate'].fillna(df.heartRate.mean(), inplace = True)

# Features and label
features = df.iloc[:,:-1]
result = df.iloc[:,-1] # the last column is what we are about to forecast

# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(features, result, test_size = 0.2, random_state = 14)

# RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.12
sfm = SelectFromModel(clf, threshold=0.12)

# Train the selector
sfm.fit(X_train, y_train)

# Features selected
featureNames = list(features.columns.values) # creating a list with features' names
print("Feature names:")
for featureNameListindex in sfm.get_support(indices=True):
    print(featureNames[featureNameListindex])

# Feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# With only imporant features. Can check X_important_train.shape[1]
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
rfc.fit(X_important_train, y_train)

preds = rfc.predict(X_important_test)

run.log("Accuracy:",metrics.accuracy_score(y_test, preds))

print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))

#joblib.dump(rfc, "/outputs/model.joblib")
os.makedirs('./outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=rfc, filename='./outputs/sklearn_diabetes_model.pkl')
```

Create the estimator to run the pipeline

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

Submit and run the experiment.

```
run = experiment.submit(estimator)
run.wait_for_completion(show_output=True)
```

Print metrics coming out of the model

```
print(run.get_metrics())
```

There should be something like {'Accuracy:': 0.8290094339622641}

To display result and progrss

```
from azureml.widgets import RunDetails
RunDetails(run).show()
```

Now it is time to hyper tune parameters to validate model

```
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, loguniform

param_sampling = GridParameterSampling( {
        "num_hidden_layers": choice(1, 2, 3),
        "batch_size": choice(16, 32)
    }
)
```

Setup parameter

```
primary_metric_name="accuracy",
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE
```

Setup the bandit policy

```
from azureml.train.hyperdrive import BanditPolicy
early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)
```

Setup Truncation Policy

```
from azureml.train.hyperdrive import TruncationSelectionPolicy
early_termination_policy = TruncationSelectionPolicy(evaluation_interval=1, truncation_percentage=20, delay_evaluation=5)
```

Setup Exit criteria for hyper tune parameters

```
max_total_runs=20,
max_concurrent_runs=4
```

Setup the Hyperparameter tuning run

```
from azureml.train.hyperdrive import HyperDriveConfig
hyperdrive_run_config = HyperDriveConfig(estimator=estimator,
                          hyperparameter_sampling=param_sampling, 
                          policy=early_termination_policy,
                          primary_metric_name="accuracy", 
                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                          max_total_runs=100,
                          max_concurrent_runs=4)
```

Submit and Run the Hyper parameter tuning

```
from azureml.core.experiment import Experiment
experiment = Experiment(workspace=ws, name='diabetes-model')
hyperdrive_run = experiment.submit(hyperdrive_run_config)
```

Display status

```
from azureml.widgets import RunDetails
RunDetails(hyperdrive_run).show()
```

Get the best performed model

```
best_run = hyperdrive_run.get_best_run_by_primary_metric()
sprint(best_run.get_details()['runDefinition']['arguments'])
```

Display model

```
print(best_run.get_file_names())
```



Now time to register model

```
# register model
model = best_run.register_model(model_name='sklearn_diabetes',
                           model_path='outputs/sklearn_diabetes_model.pkl',
                           tags=run.get_metrics())
print(model.name, model.id, model.version, sep='\t')
```

Best model will be registered. Now the model can be used for Deployment.