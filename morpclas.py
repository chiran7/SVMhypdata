import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# from scipy.misc import imshow
# from scipy.misc import toimage
from pylab import *
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import cv2
import pandas as pd



data = scipy.io.loadmat('data\\PaviaU.mat')
X = data['paviaU']
print('dimention of data',X.shape)



T = scipy.io.loadmat('data\\PaviaU_gt.mat')
T= T['paviaU_gt']
print("gt.shape",T.shape)
Tf = T.flatten()
print("Tf.shape", Tf.shape)
#Xr= np.resize(X, (145*145,200))
Xr= np.resize(X, (610*340,103))
print(Xr.shape)
pca = PCA(.99)
Xpc = pca.fit(Xr)
#print(Xpc)
print(pca.n_components_)
print(pca.components_)

fpc = pca.components_[0]
spc = pca.components_[1]
tpc = pca.components_[2]
frpc = pca.components_[3]

Xm=np.mean(Xr, axis=1)
#Xm = Xm.reshape(-1, 1)
print(Xm.shape)



npc = pca.n_components_
npx = 2
# print(pca.n_components_)
s = np.ones((5,5),np.uint8)



images_gt = np.array(Tf)
images = np.array(Xr)

openv = cv2.morphologyEx(images, cv2.MORPH_OPEN, s)
print("opening")
print(openv.shape,openv)
#closing = cv2.morphologyEx(openv, cv2.MORPH_CLOSE, s)
closing = cv2.morphologyEx(openv, cv2.MORPH_CLOSE, s)
print("closing")
print(closing.shape,closing)
print(s)

print(images.shape)
print(images_gt.shape)



trnX, tstX, trnT, tstT = train_test_split(closing,images_gt,test_size=0.95, random_state=5)
print(trnX.shape,tstX.shape,trnT.shape,tstT.shape)




model = SVC(kernel='rbf', C=1, gamma=1)
model.fit(trnX, trnT)
print("predicting")
#predicted= model.predict(tstX)
print(model.score(trnX, trnT))
print(tstX[:20000,:].shape,tstT[:20000].shape)
print('Accuracy score of SVM after morphological profile',model.score(tstX[:20000,:],tstT[:20000]))




