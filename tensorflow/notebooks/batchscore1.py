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

# +
import json
import numpy as np
import os
import tensorflow as tf

from azureml.core.model import Model
from azureml.core import Run
from azureml.core.dataset import Dataset

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

def init():
    global X, output, sess
    tf.reset_default_graph()
    model_root = os.getenv('AZUREML_MODEL_DIR')
    
    model = Model(ws, 'tf-dnn-mnist')
    
    # the name of the folder in which to look for tensorflow model files
    tf_model_folder = 'model'
    #saver = tf.train.import_meta_graph(
    #    os.path.join(model_root, tf_model_folder, 'mnist-tf.model.meta'))
    saver = tf.train.import_meta_graph(Model.get_model_path('tf-dnn-mnist',5,ws))
    X = tf.get_default_graph().get_tensor_by_name("network/X:0")
    output = tf.get_default_graph().get_tensor_by_name("network/output/MatMul:0")

    sess = tf.Session()
    saver.restore(sess, os.path.join(model_root, tf_model_folder, 'tf-dnn-mnist.model'))


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    out = output.eval(session=sess, feed_dict={X: data})
    y_hat = np.argmax(out, axis=1)
    return y_hat.tolist()
