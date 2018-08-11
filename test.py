import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

data = scipy.io.loadmat('data\\PaviaU.mat')
X = data['paviaU']
print('dimention of data',X.shape)


T = scipy.io.loadmat('data\\PaviaU_gt.mat')
T= T['paviaU_gt']
print("gt.shape",T.shape)
Tf = T.flatten()
print("Tf.shape", Tf.shape)
#Xr= np.resize(X, (145*145,200))
#Xr= np.resize(X, (1096*715,102))
Xr= np.resize(X, (610*340,103))

print(Xr.shape)

# pca = PCA(.99)
# Xpc = pca.fit(Xr)
# print(Xpc)


Xm=np.mean(Xr, axis=1)
print(Xm.shape)

trnX, tstX, trnT, tstT = train_test_split(Xr,Tf,test_size=0.99, random_state=5)
print('shape of training data, training label, test data, test label \n',trnX.shape,tstX.shape,trnT.shape,tstT.shape)


model = SVC(kernel='poly', C=1, gamma=2, degree=3)
model.fit(trnX, trnT)
print("\n predicting")
#predicted= model.predict(tstX)
print('\n Accuracy of SVM classifier on training set:',model.score(trnX, trnT))
print(tstX[:20000,:].shape,tstT[:20000].shape)
print('\n Accuracy of SVM classifier on test set:',model.score(tstX[:20000,:], tstT[:20000]))


svm = SVC(kernel='rbf', C=1, gamma=1)
svm.fit(trnX, trnT)
print("predicting")
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(trnX, trnT)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(tstX, tstT)))



