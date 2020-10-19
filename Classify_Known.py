# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 02:14:57 2020

@author: jitup
"""

from data_preparation import split,load_dataset
# from numpy import savez_compressed
# from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# path_to_data=r'C:\Users\jitup\Desktop\Nirovision\Verify\faces'
# path_to_test_data=r'C:\Users\jitup\Desktop\Nirovision\Verify\Data'
# train_ratio=0.8

# split(path_to_data, path_to_test_data, train_ratio)

# load train dataset
trainX, trainy = load_dataset('Data/train/')
labels = trainy
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('Data/val/')

# save arrays to one file in compressed format
# savez_compressed('faces-embeddings.npz', trainX, trainy, testX, testy)
# data = load('faces-embeddings.npz')
# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit svm model to Classify a face as one of a known set of people.
svm = SVC(kernel='linear', probability=True)
svm.fit(trainX, trainy)
#predict
example_prediction_svm = svm.predict(testX)
example_identity_svm = out_encoder.inverse_transform(example_prediction_svm)
#Score/accuracy
acc_svm = accuracy_score(testy, svm.predict(testX))
print(acc_svm)
    
#fit knn model to Classify a face as one of a known set of people.
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(trainX, trainy)
example_prediction_knn = knn.predict(testX)
example_identity_knn = out_encoder.inverse_transform(example_prediction_knn)
acc_knn = accuracy_score(testy, knn.predict(testX))
print(acc_knn)