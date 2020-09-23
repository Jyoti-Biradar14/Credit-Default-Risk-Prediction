#!/usr/bin/env python
# coding: utf-8

# # $ Credit\ Risk\ Default\ Prediction\ Using\ Machine\ Learning $

# 

# 

# ## Import necessory libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Load train and test file

# In[2]:


trainD = pd.read_csv("epsilon_train.csv")
trainD = pd.DataFrame(trainD)
testD = pd.read_csv("epsilon_test.csv")
testD = pd.DataFrame(testD)
trainD.head()


# In[ ]:


# Train datatype info
trainD.info()


# In[ ]:


# Check columnwise null values
print(trainD.isnull().sum())


# ## Handling the missing values (Fill with appropriate mean, mode or median value)

# In[ ]:


# Default column 
trainD.Default.value_counts(sort=True) 
trainD.Default.fillna('Yes',inplace=True)

#Checking amount column 
trainD.Checking_amount.value_counts(sort=True) 
trainD.Checking_amount.fillna(trainD["Checking_amount"].mean(),inplace=True) 

 #Term column 
trainD.Term.value_counts(sort=True) 
trainD.Term.fillna(trainD["Term"].mean(),inplace=True) 

# credit score column 
trainD.Credit_score.value_counts(sort=True) 
trainD.Credit_score.fillna(trainD["Credit_score"].mean(),inplace=True) 

# Car_loan column 
trainD.Car_loan.value_counts(sort=True) 
trainD.Car_loan.fillna("Yes",inplace=True) 

# personal loan column 
trainD.Personal_loan.value_counts(sort=True) 
trainD.Personal_loan.fillna('Yes',inplace=True)

# home loan column 
trainD.Home_loan.value_counts(sort=True) 
trainD.Home_loan.fillna('Yes',inplace=True) 

# Education loan column 
trainD.Education_loan.value_counts(sort=True) 
trainD.Education_loan.fillna('Yes',inplace=True) 

# amount 
trainD.Amount.value_counts(sort=True) 
trainD.Amount.fillna(trainD["Amount"].mean(),inplace=True) 

# Emp_duration 
trainD.iloc[:,14].value_counts(sort=True) 
trainD.iloc[:,14].fillna(trainD.iloc[:,14].mean(),inplace=True) 

#age   
trainD.Age.value_counts(sort=True) 
trainD.Age.fillna(trainD["Age"].mean(),inplace=True) 

#No_of_credit_acc column 
trainD.No_of_credit_acc.value_counts(sort=True) 
trainD.No_of_credit_acc.fillna(1,inplace=True)


# In[ ]:


# Drop all the rows which contains null values
# trainD = trainD.dropna(axis = 0, how ='any')


# In[ ]:


# check whether all null values are removed or not
print(trainD.isnull().sum())


# In[ ]:


testD.head()


# In[ ]:


testD.info()


# In[ ]:


print(testD.isnull().sum())


# In[ ]:


testD=testD.dropna(axis=0,how='any')


# In[ ]:


print(testD.isnull().sum())


# # Label Encoding of Categorical Variables

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


number=LabelEncoder()
trainD['Gender']=number.fit_transform(trainD['Gender'].astype('str'))
testD['Gender']=number.fit_transform(testD['Gender'].astype('str'))


# In[ ]:


trainD.head()


# In[ ]:


testD.head()


# In[ ]:


#labelEncoding for on more variables
number=LabelEncoder()
trainD['Personal_loan']=number.fit_transform(trainD['Personal_loan'].astype('str'))
testD['Personal_loan']=number.fit_transform(testD['Personal_loan'].astype('str'))


# In[ ]:


trainD.head()


# In[ ]:


trainD['Emp_status']=number.fit_transform(trainD['Emp_status'].astype('str'))
testD['Emp_status']=number.fit_transform(testD['Emp_status'].astype('str'))


# In[ ]:


trainD.head()


# In[ ]:


trainD['Default']=number.fit_transform(trainD['Default'].astype('str'))


# In[ ]:


trainD['Car_loan']=number.fit_transform(trainD['Car_loan'].astype('str'))
testD['Car_loan']=number.fit_transform(testD['Car_loan'].astype('str'))


# In[ ]:


trainD['Home_loan']=number.fit_transform(trainD['Home_loan'].astype('str'))
testD['Home_loan']=number.fit_transform(testD['Home_loan'].astype('str'))


# In[ ]:


trainD['Education_loan']=number.fit_transform(trainD['Education_loan'].astype('str'))
testD['Education_loan']=number.fit_transform(testD['Education_loan'].astype('str'))


# In[ ]:


trainD['Marital_status']=number.fit_transform(trainD['Marital_status'].astype('str'))
testD['Marital_status']=number.fit_transform(testD['Marital_status'].astype('str'))


# In[27]:


trainD.head()


# ## Separating the Feature Variables and Target Variable

# In[28]:


X=trainD[['ID','Checking_amount','Term','Credit_score','Gender','Marital_status','Car_loan','Personal_loan','Home_loan','Education_loan','Emp_status','Amount','Saving_amount','Emp_duration','Age','No_of_credit_acc']]


# In[29]:


y=trainD['Default']


# ## Split the train data into train-validation data

# In[30]:


#split X and y into training and testing datasets
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.25,random_state=0)


# # 1. Logistic Regression

# In[31]:


#import classes
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()


# ## Train the model

# In[32]:


lr = model1.fit(X_train,y_train)


# ## predict on validation data to check accuracy

# In[33]:


pred = lr.predict(X_val)


# ## Results of LR

# In[34]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_val,pred))
print("Precision:",metrics.precision_score(y_val,pred))
print("Recall:",metrics.recall_score(y_val,pred))


# In[35]:


# Confusion Matrix
cnf_matrix=metrics.confusion_matrix(y_val,pred)
cnf_matrix


# ## Prediction on Test Data(predict the values of default column) using LR

# In[36]:


# Create a submission file 
submission1 = pd.DataFrame()
testD1 = testD.copy()

testD1['Default'] = lr.predict(testD1)

submission1['ID'] = testD1['ID']
submission1['Default'] = testD1['Default']


# Submission file to csv
submission1.to_csv('LR_Predictions.csv', index=False)
submission1.head()


# In[ ]:





# # 2. Random Forest

# In[37]:


from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators=1000,random_state=20)

rf = model2.fit(X_train,y_train)


# In[38]:


prediction_RF = rf.predict(X_val)


# In[39]:


from sklearn import metrics

print("Accuracy=",metrics.accuracy_score(y_val, prediction_RF))
print("Precision:",metrics.precision_score(y_val,prediction_RF))
print("Recall:",metrics.recall_score(y_val,prediction_RF))


# In[40]:


# Confusion Matrix
cnf_matrix_RF =metrics.confusion_matrix(y_val,prediction_RF)
cnf_matrix_RF


# ## Prediction on test data using Random Forest

# In[41]:


# Create a submission file 
submission2 = pd.DataFrame()
testD2 = testD.copy()

testD2['Default'] = rf.predict(testD2)


submission2['ID'] = testD2['ID']
submission2['Default'] = testD2['Default']


# Submission file to csv
submission2.to_csv('RF_Predictions.csv', index=False)
submission2.head()


# # Happy Learning !!

# .
