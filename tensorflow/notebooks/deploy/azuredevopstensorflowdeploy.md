# Azure DevOps Train tensorflow model and Register

## Use Azure DevOps Azure machine learning service connection.

Ability to CI/CD using Azure DevOps and doing training. Create a training script first and then use the batch train script.

## Requirements

- Azure DevOps Account
- Github Repo with training and train file.
- create training file to retrain the model - tf_mnist.
- need agent-dependencies.sh file for all dependant libraries to install

## Architecture

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/tfmlopsflowbuild.jpg "Architecture")

## Steps to Build - Machine learning Training

Content of agent-dependencies.sh file 

```
python --version
pip install azure-cli==2.0.72
pip install --upgrade azureml-sdk
pip install --upgrade azureml-sdk[cli]
pip install tensorflow
```

Tensorflow version i saw was 2.2.0

Now time to go to https://dev.azure.com

Login with your account information

On the left menu go to Pipelines and create a new pipeline. We don't need to create a repo.

For Code changes you can clone the repo and use Visual Studio to build the scripts. Or Use the Jupyter lab and connect to repo and build the scripts.

https://github.com/balakreshnan/mlops

1. Create a New Pipeline

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf0.jpg "DevOps")

2. Select classic editor

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf1.jpg "DevOps")

3. Select the Github Repo to connect

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf2.jpg "DevOps")

4. Create Variables

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf12.jpg "DevOps")

Also please create a Machine learning service connection. (note: might have to install azure machine learning extension)

5. Create a Empty Job

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf3.jpg "DevOps")

6. Configure agent configuration for example in our case use ubuntu 16.04 as linux version

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf4.jpg "DevOps")

7. Lets setup python version

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf5.jpg "DevOps")

8. Install dependencies

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf6.jpg "DevOps")

9. Copy files to build folder

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf7.jpg "DevOps")

10. Run the Pipeline config python file to start the train.

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf8.jpg "DevOps")

11. Save and queue the job to run.

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf9.jpg "DevOps")

12. Examine the logs for any error and troubleshoot.

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf10.jpg "DevOps")

If the job run succeeds then log into azure machine learning service workspace UI and check the experiment link.

Click on the experiment and check the latest run. Go to output/logs and check the logs

Click on Model on the left menu and see if the model got registered.

Above experiment is tensorflow so tensorflow graph will be saved,

Below is the picture how artifacts should look

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/devops-mlopstf11.jpg "DevOps")