# Data Science Intro

## Machine learning using scikit regression using Azure Machine Learning services
In this intro, you will learn how to: 
- Load a dataset and do some basic data wrangling
- Build and run a linear regression model
- Write to Azure SQL Database from the notebook
- Access the Azure SQL Database in Power BI to create visualizations

## Prerequisites

- Azure Account
- Create a resource group
- Create Azure Machine Learning resource
- Create a compute instance in Azure Machine Learning
- Create Azure sql database 
- Create sample tables

## Install mssql odbc drivers for python

- The code you will run below is based on this link - https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15#ubuntu17
- Open Jupyter lab in your compute instance and then open new terminal
- In the terminal, determine your OS version by running: 
```
cat /etc/lsb-release
```
- Copy each block separately and then run in the terminal window

```
sudo su
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
```

```
#Download appropriate package for the OS version
#Choose only ONE of the following, corresponding to your OS version

#Ubuntu 16.04
curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

#Ubuntu 18.04
curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

#Ubuntu 20.04
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
```
```
exit
````
```
sudo apt-get update
```
```
sudo ACCEPT_EULA=Y apt-get install msodbcsql17
```
```
# optional: for bcp and sqlcmd
sudo ACCEPT_EULA=Y apt-get install mssql-tools
```
```
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile
```
```
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
```
```
source ~/.bashrc
```
```
# optional: for unixODBC development headers
sudo apt-get install unixodbc-dev
```

- Once this is run restart the kernel before you start the jupyter programming

## Data Science process

- Let's load sample data set

```
from azureml.opendatasets import NycTlcYellow
from dateutil import parser

end_date = parser.parse('2019-06-06')
start_date = parser.parse('2018-05-01')
nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
nyc_tlc_df = nyc_tlc.to_pandas_dataframe()
```

- Let's make sure there are data available

```
display(nyc_tlc_df.head(5))
```

- Let's analyze the column names

```
nyc_tlc_df.columns
```

- Based on the data set let's do some group by

```
from pandas import *
nyc_tlc_df.groupby(['fareAmount']).mean()
```

- Let's describe the dataset

```
nyc_tlc_df.describe()
```

- Display the shape

```
nyc_tlc_df.shape
```

- Display memory

```
nyc_tlc_df.memory_usage()
```

- Display and validate data columns

```
nyc_tlc_df["tpepPickupDateTime"]
```

- Let's create new column (ETL) for year, month and day

```
# Create new columns
nyc_tlc_df['day'] = nyc_tlc_df['tpepPickupDateTime'].dt.day
nyc_tlc_df['month'] = nyc_tlc_df['tpepPickupDateTime'].dt.month
nyc_tlc_df['year'] = nyc_tlc_df['tpepPickupDateTime'].dt.year
display(nyc_tlc_df)
```

- Display the dataset and make sure the new columns year, month and day are created

- Let's try differnt ways of grouping which is also called wrangling

```
grouped_multiple = nyc_tlc_df.groupby(['year', 'month', 'day']).agg({'fareAmount': ['mean', 'min', 'max'], 'totalAmount': ['mean', 'min', 'max']})
grouped_multiple = nyc_tlc_df.groupby(['year', 'month', 'day']).agg({'fareAmount': ['sum', 'count', 'max']})
grouped_multiple = nyc_tlc_df.groupby(['year', 'month', 'day']).agg({'fareAmount': ['sum']})
grouped_multiple = grouped_multiple.reset_index()
print(grouped_multiple)
```

- Grouped_mulitple is the data set we are going to use for machine learning
- I am choosing linear regression to show how to predict future
- By no means this is a production ready nor great model
- Intent here is show the process and commands
- Let's import the necessay libraries for modelling and plotting

```
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
%matplotlib inline
```

- Now lets plot few charts to analyze the data

```
fig, ax = plt.subplots(figsize=(15,7))
grouped_multiple.groupby(['year','month']).count()['fareAmount'].plot(ax=ax)
```

```
fig, ax = plt.subplots(figsize=(15,7))
grouped_multiple.groupby(['year','month']).count()['fareAmount'].unstack().plot(ax=ax)
```

```
grouped_multiple.plot(figsize=(18,5))
```

- Anlayze the data types of new data set

```
grouped_multiple.dtypes
```

- Plot histogram for label and features

```
grouped_multiple.hist()
```

## Model sample 1

```
feature_cols = ['year', 'month', 'day']

X = pd.DataFrame(grouped_multiple['year'])
#X = grouped_multiple.iloc[:, 1:].values
#X = grouped_multiple.loc[:, feature_cols]
y = pd.DataFrame(grouped_multiple['fareAmount'])
model = LinearRegression()
scores = []

# split the records into 3 folds and train 3 times the model, 
# test and get the score of each training
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X, y)):
  model.fit(X.iloc[train,:], y.iloc[train,:])
  score = model.score(X.iloc[test,:], y.iloc[test,:])
  scores.append(score)
print(scores)
```

```
X['tod'] = grouped_multiple['year']
#X['tod'] = grouped_multiple.loc[:, feature_cols]
# drop_first = True removes multi-collinearity
add_var = pd.get_dummies(X['tod'], prefix='tod', drop_first=True)
# Add all the columns to the model data
X = X.join(add_var)
# Drop the original column that was expanded
X.drop(columns=['tod'], inplace=True)
print(X.head())
```

```
model = LinearRegression()
scores = []
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X, y)):
 model.fit(X.iloc[train,:], y.iloc[train,:])
 scores.append(model.score(X.iloc[test,:], y.iloc[test,:]))
print(scores)
```

## model sample 2

```
feature_cols = ['year', 'month', 'day']

#X = pd.DataFrame(grouped_multiple['year'])
#X = grouped_multiple.iloc[:, 1:].values
X = grouped_multiple.loc[:, feature_cols]
y = pd.DataFrame(grouped_multiple['fareAmount'])
model = LinearRegression()
scores = []
```

- split the data set

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

- create the model

```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

- print outputs

```
print(regressor.intercept_)
print(regressor.coef_)
```

- Predict output 

```
y_pred = regressor.predict(X_test)
y_pred
```

- Print predicted output coefficients

```
y_pred = regressor.intercept_ + np.sum(regressor.coef_ * X, axis=1)
print('predicted response:', y_pred, sep='\n')
```

- Lets print the output metrics to validate and see how the model performed

```
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

- Print Coeffient

```
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df
```

## Save data to database

- now lets access Azure SQL database from notebook to test and visualize
- below code is a sample
- please make sure proper names are replaced for your Azure SQL database resource.
- Idea here is sometime it might be easy to write to Azure SQL and then use Power BI to visualize

First, install the pyodbc module
```
!pip install pyodbc
```

```
import pyodbc
server = 'servername.database.windows.net'
database = 'dbname'
username = 'adminuser'
password = 'password1'   
driver= '{ODBC Driver 17 for SQL Server}'

with pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
    with conn.cursor() as cursor:
        cursor.execute("select * from tablename")
        row = cursor.fetchone()
        while row:
            print (str(row[0]) + " " + str(row[1]))
            row = cursor.fetchone()
```
*Note: You may need to modify your server's firewall rules to execute the code above successfully  

## Connect to Power BI
- To begin building data vizualizations, open the desktop version of Power BI
- Click **Get Data**
- Under "Azure," select Azure SQL Database
- Enter the server name which should be in the format `servername.database.windows.net`
- For this example, we will use the Import mode, but DirectQuery is also available for large volumes of data
- On the next screen, enter your username and password for the database
- Modify the data using the Power Query Editor, and once completed load the data
- Build your visualizations

- Have fun
