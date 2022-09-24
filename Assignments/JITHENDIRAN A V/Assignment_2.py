#1.Importing the Requried Package

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

#2.Loading the Dataset

df = pd.read_csv("Churn_Modelling.csv")
df

#3. Visualizations

#3.1 Univariate Analysis

sns.displot(df.Gender)

#3.2 Bi-Variate Analysis

df.plot.line()

#3.3 Multi - Variate Analysis

sns.lmplot("Tenure","NumOfProducts",df,hue="NumOfProducts", fit_reg=False);

#4.Perform descriptive statistics on the dataset.

df.describe()

#5.Handle the Missing values.

data = pd.read_csv("Churn_Modelling.csv")
pd.isnull(data["Gender"])

#6.Find the outliers and replace the outliers.

df["Age"] = np.where(df["Age"] >75, median,df['Age'])
df["Age"]

#7.Check for Categorical columns and perform encoding.

pd.get_dummies(df, columns=["Gender", "Age"], prefix=["Age", "Gender"]).head()

#8.Split the data into dependent and independent variables.

#8.1 Split the data into Independent variables.

X = df.iloc[:, :-1].values
print(X)

#8.2 Split the data into Dependent variables.

Y = df.iloc[:, -1].values
print(Y)

#9.Scale the independent variables.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[["CustomerId"]] = scaler.fit_transform(df[["CustomerId"]])  
print(df)

#10.Split the data into training and testing.

from sklearn.model_selection import train_test_split
train_size=0.8
X = df.drop(columns = ['Tenure']).copy()
y = df['Tenure']
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)