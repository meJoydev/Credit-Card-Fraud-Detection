#!/usr/bin/env python
# coding: utf-8

# # IMPORTING ALL THE REQUIRED DEPENDENCIES

# In[104]:


import numpy as np
import pandas as pd
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
# MODELS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# # IMPORTING DATASET INTO PANDAS DATAFRAME

# In[105]:


credit_card_data=pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Documents\\creditcard.csv")


# # SHAPE OF THE DATASET

# In[106]:


credit_card_data.shape


# # FIRST FIVE ROWS OF DATASET

# In[107]:


credit_card_data.head()


# # LAST FIVE ROWS OF DATASET

# In[108]:


credit_card_data.tail()


# # INFORMATION ABOUT DATASET

# In[109]:


credit_card_data["Amount"].describe()


# In[110]:


credit_card_data.info()


# # CHECKING MISSING VALUE IN THE DATASET

# In[111]:


credit_card_data.isnull().sum()


# # DISTRUBUTIONN OF LEGIT AND FRAUDULENT TRANSACTION

# In[112]:


credit_card_data['Class'].value_counts()


# In[113]:


legit=len(credit_card_data[credit_card_data.Class==0])
fraud=len(credit_card_data[credit_card_data.Class==1])


# In[114]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
transaction=['legit','fraud']
count=[legit,fraud]
ax.bar(transaction,count)
plt.xlabel('transaction')
plt.ylabel('counts')
plt.title('fraud detection')
plt.show()


# In[115]:


# THIS DATASET IS HIGHLY UNBALANCED HENCE ITS HARDER FOR OUR MODEL TO COMPARE BECAUSE ONE PART CONSISTS OF MORE THAN 90% OF DATA
# SO IT WILL JUDGE ON THE BASIC OF MAJORITY DATA BUT NOT THE MINORITY DATA


# # SEPERATING THE DATA FOR ANALYSIS

# In[116]:


legit = credit_card_data[credit_card_data.Class==0]
fraud = credit_card_data[credit_card_data.Class==1]


# # SHAPE OF LEGIT & FRAUD DATA

# In[117]:


legit.shape


# In[118]:


fraud.shape


# # STATISTICAL MEASUREMENT OF DATA

# In[119]:


legit.Amount.describe()


# In[120]:


fraud.Amount.describe()


# # SPLITTING DATA INTO TRAINING AND TESTING DATA

# In[121]:


X = credit_card_data.drop(columns='Class',axis=1)
Y = credit_card_data['Class']


# In[122]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# # MODEL BUILDING :

# # RANDOM FOREST CLASSIFIER

# In[123]:


model = RandomForestClassifier()
model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)


# In[124]:


n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is Random Forest Classifier")
  
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))
  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))
  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))
  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[125]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : Random Forest Classifier")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # LOGISTIC REGRESSION

# In[126]:


LRmodel = LogisticRegression()
LRmodel.fit(X_train,Y_train)
Y_predict = LRmodel.predict(X_test)


# In[127]:


n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is Logistic Regression")
  
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))
  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))
  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))
  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[128]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : Logistic Regression")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # K NEIGHBOUR CLASSIFIER

# In[129]:


KNNmodel = KNeighborsClassifier()
KNNmodel.fit(X_train,Y_train)
Y_predict = KNNmodel.predict(X_test)


# In[130]:


n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is K Neighbors Classifier")
  
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))
  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))
  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))
  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[131]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : K Neighbors Classifier")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # DECISION TREE CLASSIFIER

# In[132]:


DTmodel = DecisionTreeClassifier()
DTmodel.fit(X_train,Y_train)
Y_predict = DTmodel.predict(X_test)


# In[133]:


n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is Decision Tree Classifier")
  
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))
  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))
  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))
  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[134]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : Decision Tree Classifier")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # OVERSAMPLING THE UNBALANCED DATA FOR BETTER RESULTS

# In[135]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[136]:


X_train_smote, Y_train_smote = smote.fit_resample(X_train.astype('float'),Y_train)


# In[137]:


print("Before smote : ", Counter(Y_train))
print("After smote : ", Counter(Y_train_smote))


# # RANDOM FOREST AFTER OVERSAMPLING

# In[138]:


model.fit(X_train_smote,Y_train_smote)
Y_predict = model.predict(X_test)

n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is Random Forest Classifier")  
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[139]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : Decision Tree Classifier")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # LOGISTIC REGRESSION AFTER OVERSAMPLING

# In[140]:


LRmodel.fit(X_train_smote,Y_train_smote)
Y_predict = LRmodel.predict(X_test)

n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is LogisticRegression")  
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[141]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : Logistic Regression")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # K NEIGHBORS CLASSIFIER AFTER OVERSAMPLING

# In[142]:


KNNmodel.fit(X_train_smote,Y_train_smote)
Y_predict = KNNmodel.predict(X_test)

n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is K Neighbors Classifier")  
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[143]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : K Neighbors Classifier")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # DECISION TREE AFTER OVERSAMPLING

# In[144]:


DTmodel.fit(X_train_smote,Y_train_smote)
Y_predict = DTmodel.predict(X_test)

n_outliers = len(fraud)
n_errors = (Y_predict != Y_test).sum()
print("The model used is Decision Tree Classifier")
acc = accuracy_score(Y_test, Y_predict)
print("The accuracy is {}".format(acc))  
prec = precision_score(Y_test, Y_predict)
print("The precision is {}".format(prec))  
rec = recall_score(Y_test, Y_predict)
print("The recall is {}".format(rec))  
f1 = f1_score(Y_test, Y_predict)
print("The F1-Score is {}".format(f1))


# In[145]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, Y_predict)
plt.figure(figsize =(9, 8))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix : Decision Tree Classifier")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# # Random Forest Classifier has the best accuracy score in both conditions before (99.96%) and after (99.95%) resampling of data.
