# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:36:37 2019

@author: nikhil
"""

import pandas as pd
import nltk
data=pd.read_csv("C:\\Users\\nikhil\\Desktop\\spam classifier\\smsspamcollection\\SMSSpamCollection",sep='\t',
                 names=["label","message"])

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


stop_words=stopwords.words("english")
ps=PorterStemmer()
lemma=WordNetLemmatizer()

list=[]
for i in range(len(data)):
    """ removing stopwords, and anything not a word in a data like dot(.),semicolon(;) etc"""
    sent=re.sub("[^a-zA-Z]"," ",data['message'][i])
    sent=sent.lower()
    sent=sent.split()
    sent=[ps.stem(word) for word in sent if word not in stop_words]
    sent=" ".join(sent)
    list.append(sent)
    
#print(list)    
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_mat_cv=cv.fit_transform(list).toarray()


y_mat_cv=pd.get_dummies(data['label'])
y_mat_cv=y_mat_cv.iloc[:,1].values


#-----------------------Train-test-split---------------------------------
from sklearn.model_selection import train_test_split
x_train_cv,x_test_cv,y_train_cv,y_test_cv=train_test_split(x_mat_cv,y_mat_cv,test_size=0.25,random_state=42)


#-----------------------Modelling----------------------------------------

#Support-vector_machine:
from sklearn.svm import SVC
sv=SVC(gamma="auto")
model=sv.fit(x_train_cv,y_train_cv)
predict_y=sv.predict(x_test_cv)     

#Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(predict_y,y_test_cv) 

#confusion-matrix
from sklearn.metrics import confusion_matrix
confusion_matrix_cv=confusion_matrix(predict_y,y_test_cv)



#Logistics Regresssion:
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model_lr=lr.fit(x_train_cv,y_train_cv)
predicted_y_lr=model_lr.predict(x_test_cv)     

#Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_lr=accuracy_score(predicted_y_lr,y_test_cv) 

#confusion-matrix
from sklearn.metrics import confusion_matrix
confusion_matrix_lr=confusion_matrix(predicted_y_lr,y_test_cv)



#Decision Tree:
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
model_decison_tree=dt.fit(x_train_cv,y_train_cv)
predicted_y_dt=model_decison_tree.predict(x_test_cv)


#Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_dt=accuracy_score(predicted_y_dt,y_test_cv) 

#confusion-matrix
from sklearn.metrics import confusion_matrix
confusion_matrix_dt=confusion_matrix(predicted_y_dt,y_test_cv)


#Random Forest:
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model_random_forest=rf.fit(x_train_cv,y_train_cv)
predicted_y_rf=model_random_forest.predict(x_test_cv)

#Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_rf=accuracy_score(predicted_y_rf,y_test_cv) 

#confusion-matrix
from sklearn.metrics import confusion_matrix
confusion_matrix_rf=confusion_matrix(predicted_y_rf,y_test_cv)



#Naive-Bayes :
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
model_mnb=mnb.fit(x_train_cv,y_train_cv)
predicted_y_mnb=model_mnb.predict(x_test_cv)



from sklearn.metrics import accuracy_score
accuracy_mnb=accuracy_score(predicted_y_mnb,y_test_cv) 


from sklearn.metrics import confusion_matrix
confusion_matrix_mnb=confusion_matrix(predicted_y_mnb,y_test_cv)