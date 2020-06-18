# Build Docker images using Azure DevOps

## Use Case

Ability to build a docker file and store in github repository and then ability to build using azure devops.

## Process

## Create Dockerfile

First and foremost create a Dockerfile. in my case i created one called Dockerfile to get linux ubuntu container.

```
FROM ubuntu:18.04
COPY . /app
RUN make /app
CMD python /app/app.py
```

## Create Azure Devops Pipeline

Log into Azure DevOps: dev.azure.com

Create a new project for the build and deploy.

Load the repo from using import feature in the Repo section of the project.

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/githubimport.jpg "Github import")

Now go to Pipeline and create a new pipeline

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/deploy1.jpg "Docker deploy")

Edit Pipeline

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/deploy2.jpg "Docker deploy")

Add 2 Tasks - search for these tasks.

Build an image

Push an Image

Configure Build an image as below:

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/deploy3.jpg "Docker deploy")

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/deploy4.jpg "Docker deploy")

Now Configure the Push an image

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/deploy5.jpg "Docker deploy")

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/deploy6.jpg "Docker deploy")

Now Save and Run Pipeline.

to Enable continous integration please use trigger section to configure.

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/deploy7.jpg "Docker deploy")