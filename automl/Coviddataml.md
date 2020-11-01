# Covid 19 Data set with Azure Automated Machine learning

## Use Case

Ability to predict recovery rate using Azure Automated Machine learning. Idea is increase productivity for data science to ability to figure out which algorithm produces the best result.

Pre Requisite

- Azure Subscription
- Create a Resource group
- Create Azure Machine learning services
- Create a Compute cluster
- Find the data set from Kaggle web site
- Here is the link for data set  - https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset?select=covid_19_data.csv
- in case if you have hard time finding the data i uploaded the sample to this repo
- https://github.com/balakreshnan/mlops/blob/master/automl/covid_19_data.csv
- Create a new data set from local data store and upload to azure machine learning services
- Once uploaded should be able to explore and see the below

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml1.jpg "Auto ML")

## Model building process

- Go to Automated ML in the left screen
- Create a New experiment

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml2.jpg "Auto ML")

- Create a New Experiment and name is "CovidData"
- Select the Target column as "Recovered"
- Select a compute cluster or create a new one to use
- I choosed - STANDARD_DS14_V2 (16 Cores, 112 GB RAM, 224 GB Disk)
- Click Next
- Select Regression here. Since we are predicting a continuous variable i am choosing regression

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml3.jpg "Auto ML")

- Leave the other setting as default
- Look at Additional Configuration Settings

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml6.jpg "Auto ML")

- Also look at View featurization settings

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml7.jpg "Auto ML")

- Configuration helps to configure the model run and validation and exit criteria
- Featurization for calculate the impact of features
- Now click Finish
- Usually it takes a long time to run this model with 8MB data set

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml8.jpg "Auto ML")

- Go to Experiment and wait for experiment to run
- Experiment will run most of the alogorithmn from Scikit library
- Once completed go the run configuration

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml4.jpg "Auto ML")

- Now go to Model section to see all the model it ran

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml5.jpg "Auto ML")

- As you can see list of model it ran and it's accuracy and time it took
- The page only display few please feel free to click next to see other pages
- Each algorithm run will have it's own Run
- Click the Run to view the output/logs for that model
- For the best model you should see View Explanation
- Click on View explanation
- View the feature and it's impact

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml9.jpg "Auto ML")

- Summary importance

![alt text](https://github.com/balakreshnan/mlops/blob/master/images/covidautoml10.jpg "Auto ML")

- The above is based on the model that it performed
- If you are ok with the model performance then you can deploy
- If need more explain please click Explain Model and let it run
- To deploy the model click deploy and follow the process to deploy as API endpoint in AKS
- System will create model image and deploy the model as rest api.
- More to come