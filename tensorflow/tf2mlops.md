# Tensorflow 2.0 - Hello World for MlOps

## Technology Used

- Azure DevOps
- Azure Machine learning services

## Requirements

- Azure DevOps
- Azure machine learning services
- Service principal Account
- Github Repo with code
- agent dependencies shell script
- Visual Studio Code to modify python code.

## Use Case

## Architecture

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/tfmlopsflowbuild1.jpg "Architecture")

## Steps - MLOps - CI/CD

- Build Pipeline
- Release Pipeline

## Build Pipeline

Make sure the code is available in github repo and azure resources are created.

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/tfmlopsflowbuild.jpg "Architecture")

Log into dev.azure.com and create a project

1) First create a new pipeline (build pipeline)

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build1.jpg "Tensorflow")

2) Select classic Editor

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build2.jpg "Tensorflow")

3) For Code repo select github to pull code from Github Repository

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build3.jpg "Tensorflow")

4) Select Empty Job

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build4.jpg "Tensorflow")

5) Congiure the agent

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build5.jpg "Tensorflow")

6) Add a task for Python version and set to 3.6

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build6.jpg "Tensorflow")

7) Add task as Bash to run the dependencies shell script

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build7.jpg "Tensorflow")

Content of shell script

```
python --version
pip install azure-cli==2.0.72
pip install --upgrade azureml-sdk
pip install azureml-sdk[notebooks]
pip install --upgrade azureml-sdk[cli]
pip install tensorflow==2.0.0
```

8) Add Task to Copy the files to Build directory to execute the code.

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build8.jpg "Tensorflow")

9) Add python task and select the correct python file to execute

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build9.jpg "Tensorflow")

10) Save and Execute and wait for it to complete

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2build10.jpg "Tensorflow")

Now time to move to release

## Release Pipeline

From the left menu select release and create a new Release pipeline

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/tfmlopsflowbuild2.jpg "Architecture")

1) Click New release Pipelines

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release1.jpg "Tensorflow")

2) Select am Empty Job to start with

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release2.jpg "Tensorflow")

3) Add artifacts first and select github to get the code repo.

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release3.jpg "Tensorflow")

4) Configure the agent to use ubuntu 16.04

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release4.jpg "Tensorflow")

5) Add Task to set python version and set to 3.6

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release5.jpg "Tensorflow")

6) Add task to run the agent dependencies 

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release6.jpg "Tensorflow")

```
Note agent dependencies files is same as the one from above for build.
```

7) Add Task to copy file to artifact folder to run the code.

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release7.jpg "Tensorflow")

8) Add Python Task to run the batch scoring script

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release8.jpg "Tensorflow")

9) Save and Run the release pipeline and wait for it to finish

![alt text](https://github.com/balakreshnan/mlops/blob/master/tensorflow/images/tf2Release10.jpg "Tensorflow")

Now this shows a complete End to End CI/CD using Azure Devops and can be automated based on criteria and also gated operations can also be include.