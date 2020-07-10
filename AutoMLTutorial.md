# Creating a Classification Model using Automated Machine Learning (AutoML)

## What is AutoML

Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.
Traditional machine learning model development is resource-intensive, requiring significant domain knowledge and time to produce and compare dozens of models. With automated machine learning, you'll accelerate the time it takes to get production-ready ML models with great ease and efficiency.
You can learn more about AutoML [here](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml). 

## When to use AutoML

There are three types of models that are best suited to AutoML:

1. Classification
1. Regression
1. Time-series forecasting

For each of these scenarios, Azure Machine Learning creates a number of pipelines in parallel that try different algorithms and parameters for you. The service iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to "fit" your data. It will stop once it hits the exit criteria defined in the experiment. The Process below shows how this works:


![Image](https://github.com/katthoma/Images/blob/master/AutoML%20How%20it%20Works.png)


## Automated ML Process
In this tutorial, you will learn how to create a classification AutoML experiment and deploy it, per the image below.

![Image](https://github.com/katthoma/Images/blob/master/AutoML%20Process.png)
 
###  Create/Select Dataset
The first step of this process will be to create an Azure Machine Learning Workspace. If you’re unfamiliar with this process, visit [this link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace#:~:text=Table%201%20%20%20%20Field%20%20,your%20user%20...%20%201%20more%20rows%20)

Once created, navigate to the “Automated ML” page via the menu on the left-hand side of the workspace. To begin the process, click “+ New Automated ML run.”

The first thing the service will prompt you to do will be to select or create a dataset. Choose an existing dataset, or if the dataset is not yet created, choose “+ Create dataset.”

Go through the process to load a dataset, specify the schema and confirm the details.

### Configure Run

After selecting or creating the dataset, you are now ready to configure the AutoML run. You can add the AutoML experiment to an existing one or create a separate experiment for your work. Similarly, you can use an existing compute or create a new one. 
For the “Target Column,” specify the column that contains the values you’d like to predict from your dataset. After filling in these inputs, click “Next.”

Based on your dataset, AutoML will likely already recommend the task type (Classification, Regression or Time-series) that best suits your data. Note that you can further customize the process by exploring additional configuration and featurization settings.
Additional configuration allows you to specify the primary metric you’d like to evaluate models on, as well as if there are any algorthms you know you do not want to use for training (blocked algorithms).
The Featurization menu allows you to tell AutoML which columns you’d like to be included in the experiment.

When you’re done with your configurations, click “Finish” to begin running the experiment.   

### Evaluate Run

Once the AutoML process is completed, you will be able to evaluate your results. Note that depending on the size and complexity of your data, as well as your compute, the run process could take from a few minutes to several hours to complete. 
Once completed, you will be able to see the “Run Overview” on the Automated ML page. In the overview, you’ll be able to see the Best Model, based on the primary metric you specified, in the right-hand corner of the screen. 
You can further explore which models AutoML evaluated in the “Models” tab. Models will be ordered by the primary metric you specified in terms of best to worst.

For more details about a particular model, click the algorithm name and then at the top of the screen, click the “Explain Model” button.

The Explain Model task can provide valuable information and details on your model, particularly which features were most important in the model’s analysis. Note that it can take several minutes for Explain Model to run before you can see results. 

### Deploy Model

After evaluating the model, the last step in the AutoML process is to deploy the model that best meets your goals. The deployment process in AutoML is simple. The first step is to click on the algorithm name in the “Models” menu of the Run. You’ll then see a new menu with an option to click “Deploy.”
Clicking “Deploy” will enable you to specify the compute type, name of the deployment, as well as any authentication you’d like to add. After you deploy, it will take some time for the run to complete, but when it does, you’ll be able to navigate to the “Endpoints” menu on the left hand side of the screen to retrieve your model’s endpoint. 

Automated ML will create a real-time endpoint that will allow you to consume your model through a REST API endpoint. 

 
