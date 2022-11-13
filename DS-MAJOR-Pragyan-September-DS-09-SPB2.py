# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:17:01 2022

@author: pragy
"""
#import streamlit as st
#---------------------------# Files

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
from pandas.plotting import andrews_curves
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
#--------------------------#
# Page layout
## Page expands to full width
#st.set_page_config(page_title='Major Project', layout = 'wide')
#--------------------------#
#Model

    
    
    
    

    #-------------------
st.write("Machine Learning (Adjust the Hyperparameters)")
st.title('Machine Learning Project')
#-------------------#File ", line 62
DF = pd.read_csv("https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/IRIS.csv")
#DF = pd.read_csv('Downloads\IRIS(1).csv')                                              
   
 #-----------------#
 #Main Panel:
#st.subheader('1. Dataset')
st.write("1. Dataset")
    
df= DF

#st.markdown('1.1. Glimpse of Dataset')
st.write('**1.1. Glimpse of Dataset**')
#st.write(df)
st.info(df.head())

#st.markdown('**1.2. Data splits**')
st.write('**1.2. Data splits**')
#st.write('EDA')
st.write("**Exploratory Data Analysis**")
columns = df.shape[1]
i=1
st.write("Displaying info about NULL Values: (Blank Boxes Denote Non Null Values)")
st.write(df.isnull())
st.write("*The sum of Null Values is:*")
st.write(df.isnull().sum())
df.dropna()
df.fillna(0)
st.write("**Outliers:**")
fig, ax = plt.subplots(2,2)



ax[0,0].boxplot(x=df.iloc[:,0],data = df)
ax[0,0].set_title("Sepal Lenght")
ax[0,1].boxplot(x=df.iloc[:,1],data = df)
ax[0,1].set_title("Sepal Width")
ax[1,0].boxplot(x=df.iloc[:,2],data = df)
ax[1,0].set_title("Petal Lenght")
ax[1,1].boxplot(x=df.iloc[:,3],data = df)
ax[1,1].set_title("Petal Width")
plt.show()
st.write("*Very Few outliers(Mostly data lying in 25-75 percentile)*")
st.pyplot(fig)

columns = df.shape[1]
st.write(f''' Columns in Data {columns}.
''')

figure, Axis = plt.subplots(2,2)
Axis[0,0].scatter(df.iloc[:,0],df.iloc[:,1])
Axis[0,0].set_title("Sepal Length vs Sepal Width")
Axis[0,1].scatter(df.iloc[:,1],df.iloc[:,2])
Axis[0,1].set_title("Sepal Width vs Petal Length")
Axis[1,0].scatter(df.iloc[:,2],df.iloc[:,3])
Axis[1,0].set_title("Sepal Width vs Petal Length")
st.pyplot(figure)

#sepx
X = df.iloc[:,0:4]
Y = df.iloc[:,-1]
#st.write('Input Variable:')
st.write("*Input Variable: *",(X.shape))
#st.info(X.shape)
st.write(list(X.columns))

#st.info(list(X.columns))
#st.write('Output Variable:')
st.write('*Output Variable: *',(Y.shape))
st.info((Y.name))
#st.info(Y.shape)

#st.info(list(Y.columns))
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0)
st.write("yDeploying Logistic Regression() model: ")
model = LogisticRegression()
model.fit(X_train,Y_train)
#Y_predict_train = model.predict(X_train)
Y_predict = model.predict(X_test)
st.write("Scaling of Model Not required done only for visualization: ")
scaler = MinMaxScaler()
x_train = scaler.fit_transform(X_train)
st.write("The scaled training data is: ")
st.write(x_train)
x_test =  scaler.fit_transform(X_test)
st.write("The scaled test data is: ")
st.write(x_test)

#st.write('The accuracy of this model is:')
st.write("The accuracy of this model is: ",(accuracy_score(Y_predict,Y_test)*100))
#st.info(accuracy_score(Y_predict_test,Y_test))
st.write("Sample Prediction with data [5.1,3.5,1.4,0.2] as sepal and petal length and widths (Expected Setosa): ")
st.write(model.predict([[5.1,3.5,1.4,0.2]]))   
st.write("Dataframe: ")
st.write(df.head())

#st.write("Andrews Plot(Each Point is represented by a unique curve)")
x = pd.plotting.andrews_curves(df,"species")
x.plot()
plt.show()
#-------------------------------#
st.write("Using Decision Tree CLassifier: ")
model2 = DecisionTreeClassifier()
model2.fit(X_train,Y_train)
Y_predict_DT = model2.predict(X_test)
st.write("Using DT CLasiifier, the predicted values of output from test data are")
st.write(Y_predict_DT)
st.write('Accuracy',accuracy_score(Y_predict_DT,Y_test)*100)
st.write("A sample prediction using data [5.2,3.4,1.5,0.25] expected Setosa: ")
st.write(model2.predict([[5.2,3.4,1.5,0.25]]))
#-------------------------------#
st.write("Using KNN classifier: ")
st.write("First Modifying Dataset:")
iris = load_iris()
df2 = pd.DataFrame(iris.data,columns=iris.feature_names)
df2['TargetValues'] = iris.target
st.write("Target Added to Data")
st.write(iris.target_names)
st.write("0-Setosa,1-versicolor,2-verginica, corresponding to the species column in original dataframe")
st.write("The new modified DataFrame:")
st.write(df2.head())
df2['Class'] = df2.TargetValues.apply(lambda x: iris.target_names[x])#adds the actual species names corresponding to the original data into Python iris data
st.write("With the addition of species column: ")
st.write(df2.head())
X1 = df2.drop(['TargetValues','Class'], axis='columns')
Y1 = df2['TargetValues']
X1_train,X1_test,Y1_train,Y1_test =train_test_split(X1,Y1,random_state=0)

#-------------------#KNN
model3 = KNeighborsClassifier(n_neighbors = 12)
model3.fit(X1,Y1)
st.write("Accuracy using KNN classifier:",(model3.score(X1_test,Y1_test)*100))

st.write("A sample Prediction using [5.1,3.5,1.4,0.4] expected 'Setosa' ", model3.predict([[4.0,4.0,3.0,2.0]]))
Y1_predict = model3.predict(X1_test)
st.write("Displaying the confusion Matrix (The diagonal entries show how many predictions were actually true for individual targets, out of diagonal are the inaccuracies):")
ConfusionMatrix = confusion_matrix(Y1_test,Y1_predict)
st.write(ConfusionMatrix)
plt.figure(figsize=(10,10))
st.write("Using Heatmap: ")
Fig, ax1 = plt.subplots(1,1)
sn.heatmap(ConfusionMatrix, annot = True)
st.write(Fig)
st.write()
plt.xlabel('Predicted Values(0-Setosa,1-versicolor,2-viriginica')
plt.ylabel('ActualValues(In the sample tested)')
st.write("The classification using KNN(Report):")
st.write(classification_report(Y1_test,Y1_predict))
#-----------------------------------#
#-----------------------------------#Clustering 
st.write("Clustering: ")
model4 = KMeans(n_clusters=3)#setosa,virginica,versicolor
model4.fit(X,Y)
Y2_predicted = model4.predict(X)
st.write("Clustering Predictions:")
st.write(Y2_predicted)
Y2 = df2.TargetValues
st.write("The Accuracy of Clustering is (It will give correct value in few reloads, due to difference in naming of clusters each time over all it is 90%): ",accuracy_score(Y2,Y2_predicted)*100)
df2['Cluster'] = Y2_predicted
st.write("For information:",df2.head())
st.write("Cluster Centres, obtained were (These are the tuples from which Eucledian dist min):")
st.write((model4.cluster_centers_))