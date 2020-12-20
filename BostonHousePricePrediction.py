#!/usr/bin/env python
# coding: utf-8

# # BOSTON HOUSE PRICE PREDICTION

# On Boston House Pricing dataset in the form of csv, regression need to be applied as the prices aren't classified into predefined classes, they are continuous values not discrete values. The problem that we are going to solve here is that given a set of features that describe a house in Boston, our machine learning model must predict the house price. To train our machine learning model with boston housing data, we will be using scikit-learn’s boston dataset. In this dataset, each row describes a boston town or suburb. There are 506 rows and 13 attributes (features) with a target column (price). CRIM per capita crime rate by town ZN proportion of residential land zoned for lots over 25,000 sq.ft. INDUS proportion of non-retail business acres per town CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) NOX nitric oxides concentration (parts per 10 million) RM average number of rooms per dwelling AGE proportion of owner-occupied units built prior to 1940 DIS weighted distances to five Boston employment centres RAD index of accessibility to radial highways TAX full-value property-tax rate per 10,000usd PTRATIO pupil-teacher ratio by town B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town LSTAT % lower status of the population Each record in the database describes a Boston suburb or town.

# In[2]:


'''Importing the libraries'''
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("All required libraries imported")


# In[3]:


'''Importing dataset'''
from sklearn.datasets import load_boston
boston=load_boston()
print("Dataset imported")


# In[4]:


# Initializing the dataframe
data = pd.DataFrame(boston.data)


# In[5]:


data.describe()


# In[6]:


#Inorder to increase readability give column names
data.columns = boston.feature_names
data.head()


# In[7]:


#Adding target variable to dataframe
data['PRICE'] = boston.target 
# Median value of owner-occupied homes in $1000s


# In[8]:


data.head()


# In[9]:


#Check the shape of dataframe
data.shape


# In[10]:


'''Check for missing values'''
data.isnull().sum()


# In[11]:


# Finding out the correlation between the features
corr = data.corr()
corr.shape


# In[12]:


# Plotting the heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')


# In[13]:


'''Spliting target variable and independent variables'''
X = data.drop(['PRICE'], axis = 1)
y = data['PRICE']


# In[14]:


'''Splitting to training and testing data'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)


# # LINEAR REGRESSION

# In[15]:


'''Training the model on the training set'''
# Import library for Linear Regression
from sklearn.linear_model import LinearRegression

# Create a Linear regressor
lm = LinearRegression()

# Train the model using the training sets 
lm.fit(X_train, y_train)


# In[16]:


# Model prediction on train data
y_pred = lm.predict(X_train)


# In[17]:


# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# 𝑅^2 : It is a measure of the linear relationship between X and Y. It is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable.
# 
# Adjusted 𝑅^2 :The adjusted R-squared compares the explanatory power of regression models that contain different numbers of predictors.
# 
# MAE : It is the mean of the absolute value of the errors. It measures the difference between two continuous variables, here actual and predicted values of y.
# 
# MSE: The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value.
# 
# RMSE: The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value.

# In[18]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[19]:


# Predicting Test data with the model
y_test_pred = lm.predict(X_test)


# In[20]:


# Model Evaluation
acc_linreg = metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[21]:


lr_trainval=lm.score(X_train,y_train)*100
lr_testval=lm.score(X_test,y_test)*100
print("Training Accuracy:",lm.score(X_train,y_train)*100)
print("Testing Accuracy:",lm.score(X_test,y_test)*100)


# Here the model evaluations scores are almost matching with that of train data. So the model is not overfitting.

# # Random Forest Regressor

# In[22]:


# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg1 = RandomForestRegressor()

# Train the model using the training sets 
reg1.fit(X_train, y_train)


# In[23]:


# Model prediction on train data
y_pred = reg1.predict(X_train)


# In[24]:


# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# In[25]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[26]:


# Predicting Test data with the model
y_test_pred = reg1.predict(X_test)


# In[27]:


# Model Evaluation
acc_rf = metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_rf)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[28]:


reg_trainval=reg1.score(X_train,y_train)*100
reg_testval=reg1.score(X_test,y_test)*100
print("Training Accuracy:",reg1.score(X_train,y_train)*100)
print("Testing Accuracy:",reg1.score(X_test,y_test)*100)


# In[29]:


pip install xgboost


# # XGBoost Regressor

# In[30]:


# Import XGBoost Regressor
from xgboost import XGBRegressor

#Create a XGBoost Regressor
reg2 = XGBRegressor()

# Train the model using the training sets 
reg2.fit(X_train, y_train)


# max_depth (int) – Maximum tree depth for base learners.
# 
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)
# 
# n_estimators (int) – Number of boosted trees to fit.
# 
# gamma (float) – Minimum loss reduction required to make a further partition on a leaf node of the tree.
# 
# min_child_weight (int) – Minimum sum of instance weight(hessian) needed in a child.
# 
# subsample (float) – Subsample ratio of the training instance.
# 
# colsample_bytree (float) – Subsample ratio of columns when constructing each tree.
# 
# objective (string or callable) – Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below).
# 
# nthread (int) – Number of parallel threads used to run xgboost. (Deprecated, please use n_jobs)
# 
# scale_pos_weight (float) – Balancing of positive and negative weights.

# In[31]:


# Model prediction on train data
y_pred = reg2.predict(X_train)


# In[32]:


# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# In[33]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[34]:


#Predicting Test data with the model
y_test_pred = reg2.predict(X_test)


# In[35]:


# Model Evaluation
acc_xgb = metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[36]:


xg_trainval=reg2.score(X_train,y_train)*100
xg_testval=reg2.score(X_test,y_test)*100
print("Training Accuracy:",reg2.score(X_train,y_train)*100)
print("Testing Accuracy:",reg2.score(X_test,y_test)*100)


# # SVM Regressor

# In[37]:


# Creating scaled set to be used in model to improve our results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[38]:


# Import SVM Regressor
from sklearn import svm

# Create a SVM Regressor
reg3 = svm.SVR()


# In[39]:


# Train the model using the training sets 
reg3.fit(X_train, y_train)


# C : float, optional (default=1.0): The penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.
# 
# kernel : string, optional (default='rbf’): kernel parameters selects the type of hyperplane used to separate the data. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed’ or a callable.
# 
# degree : int, optional (default=3): Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
# 
# gamma : float, optional (default='auto’): It is for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set. Current default is 'auto' which uses 1 / n_features.
# 
# coef0 : float, optional (default=0.0): Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
# 
# shrinking : boolean, optional (default=True): Whether to use the shrinking heuristic.

# In[40]:


# Model prediction on train data
y_pred = reg3.predict(X_train)


# In[41]:


# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# In[42]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[43]:


# Predicting Test data with the model
y_test_pred = reg3.predict(X_test)


# In[44]:


# Model Evaluation
acc_svm = metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_svm)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[47]:


svm_trainval=reg3.score(X_train,y_train)*100
svm_testval=reg3.score(X_test,y_test)*100
print("Training Accuracy:",reg3.score(X_train,y_train)*100)
print("Testing Accuracy:",reg3.score(X_test,y_test)*100)


# # Evaluation and comparision of all the models

# In[48]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines'],
    'R-squared Score': [acc_linreg*100, acc_rf*100, acc_xgb*100, acc_svm*100],
    'Training Accuracy':[lr_trainval,reg_trainval,xg_trainval,svm_trainval],
    'Testing Accuracy':[lr_testval,reg_testval,xg_testval,svm_testval]})
models.sort_values(by='R-squared Score', ascending=False)


# In[49]:


data.head()


# # By comparing the Training and Testing accuracy and R-Squared Score value the best suitable model is XGBoost!
