# Importing Necessary Libraries
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for data visualization
import plotly.express as px # for plotting
import plotly.graph_objects as go # for plotting
import warnings # for ignoring warnings
warnings.filterwarnings('ignore') # for filtering warnings
import missingno as msno # for missing value visualization
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.preprocessing import StandardScaler # for feature scaling
from sklearn.linear_model import LogisticRegression # for classification
import sklearn # for one hot encoding
import streamlit as st # for web app

# load the dataset
df = pd.read_csv('./ibid_2020.csv') # for loading the dataset
st.title('Ibid Dataset Analysis') # for title
st.subheader('Data Overview') # for data overview
pd.set_option('display.max_rows', None) # for showing all rows
pd.set_option('display.max_columns', None) # for showing all columns
st.write(df.head(n=10)) # for showing the first 10 rows

# missing value visualization using missingno
st.subheader('Missing Value Visualization') # for subheader of the missing value visualization
fig, ax = plt.subplots(figsize=(10,5)) # for setting the figure size
msno.matrix(df, ax=ax) # for missing value visualization
st.pyplot(fig) # for showing the plot
# drop the missing values
df.dropna(inplace=True) # for dropping the missing values

# classification
X = df[['CHILD_GENDER','MOTHER_AGE_GRP']] # for feature variables
y = df['LOW_BIRTH_WEIGHT'] # for target variable

# one hot encoding
X= pd.get_dummies(X, drop_first=True) # for one hot encoding
y= pd.get_dummies(y, drop_first=True) # for one hot encoding
# Reshape y variable
y = y.iloc[:, 1] # Assuming the second column is the target variable

# feature scaling
scaler = StandardScaler() # for standard scaling
X = scaler.fit_transform(X) # for feature scaling

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # for splitting the data

# model building
model = LogisticRegression() # for model building
model.fit(X_train, y_train) # for model training
y_pred = model.predict(X_test) # for model prediction

# model evaluation
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred) # for accuracy
precision = sklearn.metrics.precision_score(y_test, y_pred) # for precision
recall = sklearn.metrics.recall_score(y_test, y_pred) # for recall
f1 = sklearn.metrics.f1_score(y_test, y_pred) # for f1 score
st.subheader('Model Evaluation') # for model evaluation
st.write('Accuracy:', accuracy) # for accuracy
st.write('Precision:', precision) # for precision
st.write('Recall:', recall) # for recall


