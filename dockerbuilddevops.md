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

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Now go to Pipeline and create a new pipeline

