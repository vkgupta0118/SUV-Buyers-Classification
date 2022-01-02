#!/usr/bin/env python
# coding: utf-8

# ## SUV buyer  classification

# # Vikash Gupta (203040031)

# ### Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#to disable all the warnings
import warnings
warnings.filterwarnings('ignore')


# ### Importing the dataset

# In[2]:


dataset = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


dataset.head()


# In[4]:


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# In[5]:


#X is a matrix
#y is a vector


# ### Splitting the dataset into the Training set and Test set

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ### Feature Scaling

# In[7]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### Defining the Confusion Matrix and Classification Report

# In[8]:


from sklearn.metrics import confusion_matrix, classification_report

def confusionMatrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    
def classification_Report(y_test, y_pred):
    class_report = classification_report(y_test, y_pred)
    print(class_report)


# ### For Visualising the results

# In[9]:


from matplotlib.colors import ListedColormap

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=9,6

def mapVisualisation(title, classifier,X,y):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


# ## Training the Logistic Regression model on the Training set

# ### (linear classifier)

# In[10]:


from sklearn.linear_model import LogisticRegression
classifier_log = LogisticRegression(random_state = 0)
classifier_log.fit(X_train, y_train)


# In[11]:


#Predicting the Test set results
y_pred_log = classifier_log.predict(X_test)


# In[12]:


#true values
y_test


# In[13]:


#predicted values
y_pred_log


# In[14]:


#find confusion matrix
confusionMatrix(y_test, y_pred_log)


# In[15]:


# find Classification report
classification_Report(y_test, y_pred_log)


# In[16]:


# Visualising the Training set results

mapVisualisation('Logistic Regression (Training set)', classifier_log,X_train,y_train)


# In[17]:


# Visualising the Test set results

mapVisualisation('Logistic Regression (Test set)', classifier_log,X_test,y_test)


# ## Training the K-NN model on the Training set

# ### (non-linear classifier)

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
classifier_knn= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)


# In[19]:


#Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)


# In[20]:


#true values
y_test


# In[21]:


#predicted values
y_pred_knn


# In[22]:


#find confusion matrix
confusionMatrix(y_test, y_pred_knn)


# In[23]:


# find Classification report
classification_Report(y_test, y_pred_knn)


# In[24]:


# Visualising the Training set results

mapVisualisation('K-NN (Training set)', classifier_knn,X_train,y_train)


# In[25]:


# Visualising the Test set results

mapVisualisation('K-NN (Test set)', classifier_knn,X_test,y_test)


# ## Training the SVM model on the Training set

# ### (linear classification)

# In[26]:


from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train, y_train)


# In[27]:


#Predicting the Test set results
y_pred_svm = classifier_svm.predict(X_test)


# In[28]:


#true values
y_test


# In[29]:


#predicted values
y_pred_svm


# In[30]:


#find confusion matrix
confusionMatrix(y_test, y_pred_svm)


# In[31]:


# find Classification report
classification_Report(y_test, y_pred_svm)


# In[32]:


# Visualising the Training set results

mapVisualisation('SVM (Training set)', classifier_svm,X_train,y_train)


# In[33]:


# Visualising the Test set results

mapVisualisation('SVM (Test set)', classifier_svm,X_test,y_test)


# ## Training the Kernel SVM model on the Training set

# ### (non-linear classifier)

# In[34]:


from sklearn.svm import SVC
classifier_ker = SVC(kernel = 'rbf', random_state = 0)
classifier_ker.fit(X_train, y_train)


# In[35]:


#Predicting the Test set results
y_pred_ker = classifier_ker.predict(X_test)


# In[36]:


#true values
y_test


# In[37]:


#predicted values
y_pred_ker


# In[38]:


#find confusion matrix
confusionMatrix(y_test, y_pred_ker)


# In[39]:


# find Classification report
classification_Report(y_test, y_pred_ker)


# In[40]:


# Visualising the Training set results

mapVisualisation('Kernel SVM (Training set)', classifier_ker,X_train,y_train)


# In[41]:


# Visualising the Test set results

mapVisualisation('Kernel SVM (Test set)', classifier_ker,X_test,y_test)


# ## Training the Decision Tree Classification model on the Training set

# ### (non-linear classifier)

# In[42]:


from sklearn.tree import DecisionTreeClassifier
classifier_dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dtc.fit(X_train, y_train)


# In[43]:


#Predicting the Test set results
y_pred_dtc = classifier_dtc.predict(X_test)


# In[44]:


#true values
y_test


# In[45]:


#predicted values
y_pred_dtc


# In[46]:


#find confusion matrix
confusionMatrix(y_test, y_pred_dtc)


# In[47]:


# find Classification report
classification_Report(y_test, y_pred_dtc)


# In[48]:


# Visualising the Training set results

mapVisualisation('Decision Tree (Training set)', classifier_dtc,X_train,y_train)


# In[49]:


# Visualising the Test set results

mapVisualisation('Decision Tree (Test set)', classifier_dtc,X_test,y_test)


# ## Training the Random Forest Classification model on the Training set

# ### (non-linear classifier)

# In[50]:


from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators =150, criterion = 'entropy', random_state = 0)
classifier_rfc.fit(X_train, y_train)


# In[51]:


#Predicting the Test set results
y_pred_rfc = classifier_rfc.predict(X_test)


# In[52]:


#true values
y_test


# In[53]:


#predicted values
y_pred_rfc


# In[54]:


#true values vs predicted values
print(np.concatenate((y_pred_rfc.reshape(len(y_pred_rfc),1), y_test.reshape(len(y_test),1)),1))


# In[55]:


#find confusion matrix
confusionMatrix(y_test, y_pred_rfc)


# In[56]:


# find Classification report
classification_Report(y_test, y_pred_rfc)


# In[57]:


# Visualising the Training set results

mapVisualisation('Random Forest (Training set)', classifier_rfc,X_train,y_train)


# In[58]:


# Visualising the Test set results

mapVisualisation('Random Forest (Test set)', classifier_rfc,X_test,y_test)


# ### Comparing the data models that fits the best

# In[59]:


# Model Accuracy, how often is the classifier correct?

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# In[60]:


# Applying k-Fold Cross Validation on Logistic Regression
accuracies = cross_val_score(estimator = classifier_log, X = X_train, y = y_train, cv = 10)
print("Accuracies:",list(accuracies*100))
print("\n")
print("Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[61]:


print('Accuracy of Logistic Regression on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred_log)*100))


# In[62]:


# Applying k-Fold Cross Validation on KNN Classifier
accuracies = cross_val_score(estimator = classifier_knn, X = X_train, y = y_train, cv = 10)
print("Accuracies:",list(accuracies*100))
print("\n")
print("Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[63]:


print('Accuracy of KNN Classifier on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred_knn)*100))


# In[64]:


# Applying k-Fold Cross Validation on SVM Classifier
accuracies = cross_val_score(estimator = classifier_svm, X = X_train, y = y_train, cv = 10)
print("Accuracies:",list(accuracies*100))
print("\n")
print("Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[65]:


print('Accuracy of SVM Classifier on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred_svm)*100))


# In[66]:


# Applying k-Fold Cross Validation on Kernel SVM Classifier
accuracies = cross_val_score(estimator = classifier_ker, X = X_train, y = y_train, cv = 10)
print("Accuracies:",list(accuracies*100))
print("\n")
print("Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[67]:


print('Accuracy of Kernel SVM Classifier on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred_ker)*100))


# In[68]:


# Applying k-Fold Cross Validation on Decision Tree Classifier
accuracies = cross_val_score(estimator = classifier_dtc, X = X_train, y = y_train, cv = 10)
print("Accuracies:",list(accuracies*100))
print("\n")
print("Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[69]:


print('Accuracy of Decision Tree Classifier on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred_dtc)*100))


# In[70]:


# Applying k-Fold Cross Validation on Random Forest Classifier
accuracies = cross_val_score(estimator = classifier_rfc, X = X_train, y = y_train, cv = 10)
print("Accuracies:",list(accuracies*100))
print("\n")
print("Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# # Thank You!
