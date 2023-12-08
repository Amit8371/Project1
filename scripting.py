# import all the lib
import pandas as pd
import numpy as np
from sklearn import preprocessing
from flask import Flask, request, Response


# In[2]:


# reading the dataset
df = pd.read_csv('dataset/Iris.csv')


# In[3]:


df.set_index("Id", inplace=True)


# In[4]:


X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


# In[5]:


le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)


# # Training the Model

# In[6]:


# Import train-test scikit learn
from sklearn.model_selection import train_test_split


# In[7]:


# Split the data for train and test
X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=100)


# In[8]:


#Create the SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)


# In[9]:


#Fit the model for the data
classifier.fit(X_train, y_train)


# In[10]:


from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_test)
print(f"Actual Label:{y_test}")
print(f"Predicted Label:{y_pred}")
print()
print(f"Accuracy on Test Data:{accuracy_score(y_test,y_pred)*100:.4f} %")


# #Hyperparameter Tuning

# In[11]:


from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

svc_model_1 = SVC(probability=True)
param_grid = {"C":np.arange(1,500),
             "kernel":["linear" ,"poly", "rbf"],
             "gamma":np.arange(0,1,0.1),
             "degree":[2,3,4]}

rscv_model = RandomizedSearchCV(svc_model_1,param_grid,cv=5)
rscv_model.fit(X_train,y_train)
rscv_model.best_estimator_


# In[12]:


rscv_model.best_score_


# In[13]:


# best score
print(f" best score :{rscv_model.best_score_}")
print()
# print parameters that give the best results
print(f"Parameters:\n {rscv_model.best_params_}")


# In[14]:


new_model = rscv_model.best_estimator_
new_model.fit(X_train,y_train)


# In[15]:


y_pred = new_model.predict(X_test)

print(f"Actual Label:{y_test}")
print(f"Predicted Label:{y_pred}")
print()
print(f"Accuracy on Test Data:{accuracy_score(y_test,y_pred)*100:.4f} %")


# # Saving our model

# In[16]:


from joblib import dump, load


# In[17]:


# Saving my model with the name new_model
dump(new_model, 'new_model.joblib')


# In[18]:


new_model.predict(np.array([[1.4, 1.2, 1.3, 0.5]]))
