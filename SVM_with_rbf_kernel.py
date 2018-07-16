#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:13:19 2018

@author: http://bit.ly/colab-google-kaggle-source
"""

import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def load_train_data():
    # Change return line related to normal,time,frequency
    x_train = np.load('/content/my_tut_drive/tut_kaggle/X_train.npy')
    
    #Compute the average over the frequency axis=1
    #return np.mean(x_train, axis=1)

    #Compute the average over the time axis=2
    return np.mean(x_train, axis=2)
    
    #20040-dimensional vector from each sample
    #return np.resize(x_train,(4500,20040))

def load_test_data():
    x_test = np.load('/content/my_tut_drive/tut_kaggle/X_test.npy')
    #Compute the average over the frequency axis=1
    #return np.mean(x_test, axis=1)

    #Compute the average over the time axis=2
    return np.mean(x_test, axis=2)
    
    #20040-dimensional vector from each sample
    #return np.resize(x_test,(1500,20040))
    
def load_train_labels():
    labels = []
    with open ('/content/my_tut_drive/tut_kaggle/y_train.csv', 'r') as fp:
        for line in fp:
            # Skip the first line:
            if "id,scene_label" in line:
                continue
            else:
                values = line.split(",")
                label = values[1]
                labels.append(label)
    return np.asarray(labels)

def load_cross_val(X_train, y_train):
    train_index = []
    test_index = []
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    with open ('/content/my_tut_drive/tut_kaggle/crossvalidation_train.csv', 'r') as fp:
        for line in fp:
            # Skip the first line:
            if "id,scene_label,set" in line:
                continue
            else:
                values = line.strip().split(',')
                if (values[2] == "train"):
                    train_index.append(values[0])
                else:
                    test_index.append(values[0])
                    
    train_index = list(map(int, train_index))
    test_index = list(map(int, test_index))
    
    for line in train_index:
        train_data.append(X_train[line,:])
        train_labels.append(y_train[line])
        
    for line in test_index:
        test_data.append(X_train[line,:])
        test_labels.append(y_train[line])
        
    return train_data, test_data, train_labels, test_labels



#if __name__ == "__main__": should be removed before running it in Colab.
    
#if __name__ == "__main__": 
    
    # main code
    
    X_train = load_train_data()
    y_str_train = load_train_labels() # It's train data with class names.
    
    # So convert the names to the number.
    le = preprocessing.LabelEncoder()
    le.fit(y_str_train)
    y_train = le.transform(y_str_train)
    

    
    X_train, X_test, y_train, y_test = load_cross_val(X_train, y_train)
    
    # SVM Classifiers' parameters. I optimized C=13 and tol=0.01, penalty and tolerance respectively      
    clf_SVC = SVC(C=13, cache_size=200, class_weight=None, decision_function_shape='ovr',
                   degree=3, gamma= 'auto', kernel='rbf',max_iter=-1, probability=False,
                   random_state=0, shrinking=True,tol=0.01, verbose=0)
    clf_SVC.fit(X_train, y_train)
    y_pred_SVC = clf_SVC.predict(X_test)
    acc_SVC = accuracy_score(y_test, y_pred_SVC)
    
        # K-Neighbors Classifier
    clf_KNN = KNeighborsClassifier()
    clf_KNN.fit(X_train, y_train)
    y_pred_KNN = clf_KNN.predict(X_test)
    acc_KNN = accuracy_score(y_test, y_pred_KNN)
    
    # Linear Discriminant Analysis
    clf_LDA = LinearDiscriminantAnalysis()
    clf_LDA.fit(X_train, y_train)
    y_pred_LDA = clf_LDA.predict(X_test)
    acc_LDA = accuracy_score(y_test, y_pred_LDA)
    
    # As you can see that, Support Vector Machine Classifier's accuracy
    # is the best. So we can use it for creating submission file.
    
    
    secret_test = load_test_data()
    
    y_pred = clf_SVC.predict(secret_test)
    labels = list(le.inverse_transform(y_pred))
    
    with open("/content/my_tut_drive/tut_kaggle/sub.csv", "w") as fp:
        fp.write("Id,Scene_label\n")
        
        for i, label in enumerate(labels):
            fp.write("%d,%s" % (i, label))