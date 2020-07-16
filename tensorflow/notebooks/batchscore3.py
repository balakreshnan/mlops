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

