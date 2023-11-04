import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



dir = 'datasets'

catagories = ['Happy' , 'Sad']

data = []

for category in catagories:
    path = os.path.join(dir , category)
    label = catagories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        draw_img = cv2.imread(imgpath,0)
        # cv2.imshow('image',draw_img)
        try:
            draw_img = cv2.resize(draw_img,(50,50))
            image = np.array(draw_img).flatten()

            data.append([image, label])
        except Exception as e:
            pass
        # break
    # break



print(len(data))


features = []
labels = []

for feature , label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features,labels,test_size = 0.50)

model = SVC(C=1, kernel='poly',gamma = 'auto')
model.fit(xtrain,ytrain)

model2=DecisionTreeClassifier(max_leaf_nodes=10,random_state=0)
model2.fit(xtrain,ytrain)

model3=KNeighborsClassifier(n_neighbors=5)
model3.fit(xtrain,ytrain)

model4=RandomForestClassifier()
model4.fit(xtrain,ytrain)

model5=BernoulliNB(binarize=True)
model5.fit(xtrain,ytrain)

model6=MultinomialNB()
model6.fit(xtrain,ytrain)


model7=GaussianNB()
model7.fit(xtrain,ytrain)


model8=LogisticRegression(solver='saga',max_iter=4000)
model8.fit(xtrain,ytrain)

print('Accuracy for SVM is : ',model.score(xtest,ytest))
print('Accuracy for DtreeCl is : ',model2.score(xtest,ytest))
print('Accuracy for KNN is : ',model3.score(xtest,ytest))
print('Accuracy for RForrest is : ',model4.score(xtest,ytest))
print('Accuracy for BernoulliNB is : ',model5.score(xtest,ytest))
print('Accuracy for MultinomialNB is : ',model6.score(xtest,ytest))
print('Accuracy for GaussianNB is : ',model7.score(xtest,ytest))
print('Accuracy for LogReg is : ',model8.score(xtest,ytest))

print('\n')

print('Report for SVM:\n')
pred=model.predict(xtest)
print(classification_report(ytest,pred))


print('Report for DtreeCl:\n')
pred2=model2.predict(xtest)
print(classification_report(ytest,pred2))


print('Report for KNN:\n')
pred3=model3.predict(xtest)
print(classification_report(ytest,pred3))


print('Report for RForrest:\n')
pred4=model4.predict(xtest)
print(classification_report(ytest,pred4))


print('Report for BernoulliNB:\n')
pred5=model5.predict(xtest)
print(classification_report(ytest,pred5))


print('Report for MultinomialNB:\n')
pred6=model6.predict(xtest)
print(classification_report(ytest,pred6))



print('Report for GaussianNB:\n')
pred7=model7.predict(xtest)
print(classification_report(ytest,pred7))



print('Report for LogReg:\n')
pred8=model8.predict(xtest)
print(classification_report(ytest,pred8))