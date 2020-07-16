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

import tensorflow as tf
print(tf.__version__)

from azureml.core import Experiment

script_folder = './tf-mnist'
os.makedirs(script_folder, exist_ok=True)

exp = Experiment(workspace=ws, name='tf-mnist')

import urllib.request

data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 'train-images-idx3-ubyte.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
                           filename=os.path.join(data_folder, 'train-labels-idx1-ubyte.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 't10k-images-idx3-ubyte.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz',
                           filename=os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz'))

from utils import load_data

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the neural network converge faster.
X_train = load_data(os.path.join(data_folder, 'train-images-idx3-ubyte.gz'), False) / np.float32(255.0)
X_test = load_data(os.path.join(data_folder, 't10k-images-idx3-ubyte.gz'), False) / np.float32(255.0)
y_train = load_data(os.path.join(data_folder, 'train-labels-idx1-ubyte.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz'), True).reshape(-1)

count = 0
sample_size = 30
#plt.figure(figsize = (16, 6))
#for i in np.random.permutation(X_train.shape[0])[:sample_size]:
#    count = count + 1
#    plt.subplot(1, sample_size, count)
#    plt.axhline('')
#    plt.axvline('')
#    plt.text(x = 10, y = -10, s = y_train[i], fontsize = 18)
#    plt.imshow(X_train[i].reshape(28, 28), cmap = plt.cm.Greys)
#plt.show()

from azureml.core.dataset import Dataset
web_paths = ['https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
             'https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
             'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
             'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz'
            ]
dataset = Dataset.File.from_files(path = web_paths)

dataset_registered = False
try:
    temp = Dataset.get_by_name(workspace = ws, name = 'mnist-dataset')
    dataset_registered = True
except:
    print("The dataset mnist-dataset is not registered in workspace yet.")

if not dataset_registered:
    dataset = dataset.register(workspace = ws,
                               name = 'mnist-dataset',
                               description='training and test dataset',
                               create_new_version=True)
# list the files referenced by dataset
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

import shutil

# the training logic is in the tf_mnist.py file.
shutil.copy('./tf_mnist2.py', script_folder)
shutil.copy('./utils.py', script_folder)

# the utils.py just helps loading data from 

with open(os.path.join(script_folder, './tf_mnist2.py'), 'r') as f:
    print(f.read())

from azureml.train.dnn import TensorFlow

script_params = {
    '--data-folder': dataset.as_named_input('mnist').as_mount(),
    '--batch-size': 64,
    '--first-layer-neurons': 256,
    '--second-layer-neurons': 128,
    '--learning-rate': 0.01
}

est = TensorFlow(source_directory=script_folder,
                 script_params=script_params,
                 compute_target=compute_target,
                 entry_script='tf_mnist2.py',
                 use_gpu=True,
                 framework_version='2.0',
                 pip_packages=['azureml-dataprep[pandas,fuse]'])

run = exp.submit(est)

from azureml.widgets import RunDetails

RunDetails(run).show()

run.wait_for_completion(show_output=True)

run.get_details()

run.get_metrics()

run.get_file_names()

os.makedirs('./imgs', exist_ok=True)
metrics = run.get_metrics()

#plt.figure(figsize = (13,5))
#plt.plot(metrics['validation_acc'], 'r-', lw=4, alpha=.6)
#plt.plot(metrics['training_acc'], 'b--', alpha=0.5)
#plt.legend(['Full evaluation set', 'Training set mini-batch'])
#plt.xlabel('epochs', fontsize=14)
#plt.ylabel('accuracy', fontsize=14)
#plt.title('Accuracy over Epochs', fontsize=16)
#run.log_image(name='acc_over_epochs.png', plot=plt)
#plt.show()

run.download_files(prefix='outputs/model', output_directory='./model', append_prefix=False)

import tensorflow as tf
imported_model = tf.saved_model.load('./model')

pred =imported_model(X_test)
y_hat = np.argmax(pred, axis=1)

# print the first 30 labels and predictions
print('labels:  \t', y_test[:30])
print('predictions:\t', y_hat[:30])

print("Accuracy on the test set:", np.average(y_hat == y_test))

from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, loguniform

ps = RandomParameterSampling(
    {
        '--batch-size': choice(32, 64, 128),
        '--first-layer-neurons': choice(16, 64, 128, 256, 512),
        '--second-layer-neurons': choice(16, 64, 256, 512),
        '--learning-rate': loguniform(-6, -1)
    }
)

est = TensorFlow(source_directory=script_folder,
                 script_params={'--data-folder': dataset.as_named_input('mnist').as_mount()},
                 compute_target=compute_target,
                 entry_script='tf_mnist2.py',
                 framework_version='2.0',
                 use_gpu=True,
                 pip_packages=['azureml-dataprep[pandas,fuse]'])

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

htc = HyperDriveConfig(estimator=est, 
                       hyperparameter_sampling=ps, 
                       policy=policy, 
                       primary_metric_name='validation_acc', 
                       primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                       max_total_runs=8,
                       max_concurrent_runs=4)

htr = exp.submit(config=htc)

RunDetails(htr).show()

htr.wait_for_completion(show_output=True)

assert(htr.get_status() == "Completed")

best_run = htr.get_best_run_by_primary_metric()

print(best_run.get_file_names())

model = best_run.register_model(model_name='tf-dnn-mnist', model_path='outputs/model')

