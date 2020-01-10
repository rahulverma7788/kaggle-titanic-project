#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import re


# In[2]:


dataset=pd.read_csv('train.csv')
testdata=pd.read_csv('test.csv')


# In[3]:


data_passenger_value = testdata[['PassengerId']]


# In[4]:


x_train = dataset[['Pclass', 'Sex', 'Age',  'Fare', 'Embarked']]
x_test = testdata[['Pclass', 'Sex', 'Age',  'Fare', 'Embarked']]
y_train=dataset[['Survived']]


# In[5]:


len(data_passenger_value)


# In[ ]:





# In[6]:


print(x_train.head(15))


# In[7]:


print(x_test.head(15))


# In[8]:


print(x_train.shape,x_test.shape)


# In[9]:


print(y_train.shape)


# In[10]:


x_train.info()


# In[11]:


x_test.info()


# In[12]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [x_train, x_test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[13]:


ports = {"male": 0, "female": 1}
data = [x_train, x_test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(ports)


# In[14]:


print(x_train.head(15))


# In[15]:


print(x_test.head(15))


# In[16]:


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer=imputer.fit(x_train[['Age']])
x_train[['Age']]=imputer.transform(x_train[['Age']])
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer=imputer.fit(x_train[['Embarked']])
x_train[['Embarked']]=imputer.transform(x_train[['Embarked']])


# In[17]:


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer=imputer.fit(x_test[['Age']])
x_test[['Age']]=imputer.transform(x_test[['Age']])


# In[18]:


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer=imputer.fit(x_test[['Embarked']])
x_test[['Embarked']]=imputer.transform(x_test[['Embarked']])
x_test.info()


# In[19]:


x_test.info()


# In[20]:


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer=imputer.fit(x_train[['Fare']])
x_train[['Fare']]=imputer.transform(x_train[['Fare']])


# In[21]:


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer=imputer.fit(x_test[['Fare']])
x_test[['Fare']]=imputer.transform(x_test[['Fare']])
x_test.info()


# In[22]:


x_train.info()


# In[23]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)


# In[ ]:





# In[24]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components = 2)
x_train=lda.fit_transform(x_train,y_train)
x_test=lda.transform(x_test)


# In[25]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
classifier.fit(x_train, y_train)


# In[26]:


y_pred = classifier.predict(x_train)
acc_decision_tree = round(classifier.score(x_train, y_train) * 100, 2)
acc_decision_tree


# In[27]:


from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators=100,                                         max_depth=100, random_state=0)
classifier1.fit(x_train, y_train)
y_pred1 = classifier1.predict(x_test)
acc_decision_tree = round(classifier1.score(x_train, y_train) * 100, 2)
acc_decision_tree
print(y_pred1)


# In[28]:


len(y_pred1)


# In[29]:


passenger_id = []
for i,row in data_passenger_value.iterrows():
    passenger_id.append(row['PassengerId'])
    
passenger_id


# In[30]:


submission = pd.DataFrame({'PassengerId':passenger_id,'Survived':y_pred1})
submission.head()


# In[31]:


filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
