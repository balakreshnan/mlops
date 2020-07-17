# Tensorflow 2.0 Training using Azure Machine learning service

## Use Case

For this article i am using Minst data set available to build a devops training script to be used in Azure devops build process

## Requirements

- minst data set
- Azure Devops acoount
- Azure Machine learning services
- Github to store the code

## Code requirements

- Training code for azure machine learning pipeline
- Tensorflow Train Code

```
Note: use this code as sample or reference because i am using all public available data set and can be changed or moved
```

## Process

First the Azure machine learning pipeline script tf2train.py and internally the script calls tf_minst.py.

First code for tensorflow train code

Create a python file and name is tf_minst2.py. This sample uses service principal account to run in production environment.

```
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import re
import tensorflow as tf
import time
import glob

from azureml.core import Run
from utils import load_data
from tensorflow.keras import Model, layers

import gzip
import numpy as np
import struct


# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]


# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__()
        # First hidden layer.
        self.h1 = layers.Dense(n_h1, activation=tf.nn.relu)
        # Second hidden layer.
        self.h2 = layers.Dense(n_h2, activation=tf.nn.relu)
        self.out = layers.Dense(n_outputs)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        if not is_training:
            # Apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


def cross_entropy_loss(y, logits):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        logits = neural_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(y, logits)

    # Variables to update, i.e. trainable variables.
    trainable_variables = neural_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


print("TensorFlow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=128,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=128,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
parser.add_argument('--resume-from', type=str, default=None,
                    help='location of the model or checkpoint files from where to resume the training')
args = parser.parse_args()

previous_model_location = args.resume_from
# You can also use environment variable to get the model/checkpoint files location
# previous_model_location = os.path.expandvars(os.getenv("AZUREML_DATAREFERENCE_MODEL_LOCATION", None))

data_folder = args.data_folder
print('Data folder:', data_folder)

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
X_train = load_data(glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'),
                              recursive=True)[0], False) / np.float32(255.0)
X_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'),
                             recursive=True)[0], False) / np.float32(255.0)
y_train = load_data(glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'),
                              recursive=True)[0], True).reshape(-1)
y_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'),
                             recursive=True)[0], True).reshape(-1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]

n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 10
learning_rate = args.learning_rate
n_epochs = 20
batch_size = args.batch_size

# Build neural network model.
neural_net = NeuralNet()

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# start an Azure ML run
run = Run.get_context()

if previous_model_location:
    # Restore variables from latest checkpoint.
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)
    checkpoint_file_path = tf.train.latest_checkpoint(previous_model_location)
    checkpoint.restore(checkpoint_file_path)
    checkpoint_filename = os.path.basename(checkpoint_file_path)
    num_found = re.search(r'\d+', checkpoint_filename)
    if num_found:
        start_epoch = int(num_found.group(0))
        print("Resuming from epoch {}".format(str(start_epoch)))

start_time = time.perf_counter()
for epoch in range(0, n_epochs):

    # randomly shuffle training set
    indices = np.random.permutation(training_set_size)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # batch index
    b_start = 0
    b_end = b_start + batch_size
    for _ in range(training_set_size // batch_size):
        # get a batch
        X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]

        # update batch index for the next batch
        b_start = b_start + batch_size
        b_end = min(b_start + batch_size, training_set_size)

        # train
        run_optimization(X_batch, y_batch)

    # evaluate training set
    pred = neural_net(X_batch, is_training=False)
    acc_train = accuracy(pred, y_batch)

    # evaluate validation set
    pred = neural_net(X_test, is_training=False)
    acc_val = accuracy(pred, y_test)

    # log accuracies
    run.log('training_acc', np.float(acc_train))
    run.log('validation_acc', np.float(acc_val))
    print(epoch, '-- Training accuracy:', acc_train, '\b Validation accuracy:', acc_val)

    # Save checkpoints in the "./outputs" folder so that they are automatically uploaded into run history.
    checkpoint_dir = './outputs/'
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)

    if epoch % 2 == 0:
        checkpoint.save(checkpoint_dir)

run.log('final_acc', np.float(acc_val))
os.makedirs('./outputs/model', exist_ok=True)

# files saved in the "./outputs" folder are automatically uploaded into run history
# this is workaround for https://github.com/tensorflow/tensorflow/issues/33913 and will be fixed once we move to >tf2.1
neural_net._set_inputs(X_train)
tf.saved_model.save(neural_net, './outputs/model/')

stop_time = time.perf_counter()
training_time = (stop_time - start_time) * 1000
print("Total time in milliseconds for training: {}".format(str(training_time)))
```

## Azure Machine learning pipeline code

Create a python code for to prep the experiment ready for azure machine learning pipeline.

The code uses service principal to run in production environment

python file name: tf2train.py

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
 
svc_pr_password = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
 
svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_password="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
 
ws = Workspace(
    subscription_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
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

#from azureml.widgets import RunDetails

#RunDetails(run).show()

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

#RunDetails(htr).show()

htr.wait_for_completion(show_output=True)

assert(htr.get_status() == "Completed")

best_run = htr.get_best_run_by_primary_metric()

print(best_run.get_file_names())

model = best_run.register_model(model_name='tf-dnn-mnist', model_path='outputs/model')

```