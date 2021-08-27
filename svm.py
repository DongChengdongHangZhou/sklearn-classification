import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import csv


def plot_dataset(X, y, axes):
    plt.plot( X[:,0][y==0], X[:,1][y==0], "bs" )
    plt.plot( X[:,0][y==1], X[:,1][y==1], "g^" )
    plt.axis( axes )
    plt.grid( True, which="both" )
    plt.xlabel(r"$x_l$")
    plt.ylabel(r"$x_2$")

def plot_predict(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid( x0s, x1s )
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict( X ).reshape( x0.shape )
    y_decision = clf.decision_function( X ).reshape( x0.shape )
    plt.contour( x0, x1, y_pred, cmap=plt.cm.winter, alpha=0.5 )
    plt.contour( x0, x1, y_decision, cmap=plt.cm.winter, alpha=0.2 )

def svm():
    data_pos = pd.read_csv('D:/Graduation_Dissertation/sklearn_classification/train/targetData.csv',header=None)
    data_neg = pd.read_csv('D:/Graduation_Dissertation/sklearn_classification/train/results.csv',header=None)
    data_pos = np.array(data_pos)
    data_neg = np.array(data_neg)
    X = np.concatenate((data_pos,data_neg),axis=0)
    y_pos = np.ones((8000,),dtype=int)
    y_neg = np.zeros((7000,),dtype=int)
    y = np.concatenate((y_pos,y_neg),axis=0)

    rbf_kernel_svm_clf = Pipeline([
                                    ("scaler", StandardScaler()),
                                    ("svm_clf", SVC(kernel="rbf"))
                                    # ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
                                ])
    plt.figure(figsize=(6,3))
    rbf_kernel_svm_clf.fit( X, y )

    data_test = pd.read_csv('D:/Graduation_Dissertation/sklearn_classification/test/results.csv',header=None)
    data_test = np.array(data_test)

    y_pred = rbf_kernel_svm_clf.predict(data_test)
    print(y_pred)
    f = open('TEST.csv','w',newline='')
    writer = csv.writer(f)
    writer.writerow(y_pred)
    # plot_dataset( X, y, [-1.5, 2.5, -1, 1.5] )
    # plot_predict( rbf_kernel_svm_clf, [-1.5, 2.5, -1, 1.5] )
    plt.show(  )

def kmeans():
    # X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
    #                 cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)
    X = np.array([[1,2,20],[1.2,2.2,21],[5,4,0],[4.8,4.1,19.5]])

    data_pos = pd.read_csv('D:/Graduation_Dissertation/sklearn_classification/train/targetData.csv',header=None)
    data_neg = pd.read_csv('D:/Graduation_Dissertation/sklearn_classification/train/fake_fingerprint.csv',header=None)
    data_pos = np.array(data_pos)
    data_neg = np.array(data_neg)
    X = np.concatenate((data_pos,data_neg),axis=0)

    kmeans = KMeans(n_clusters=2, random_state=9).fit(X)
    y_pred = kmeans.fit_predict(X)
    f = open('TEST.csv','w',newline='')
    writer = csv.writer(f)
    writer.writerow(y_pred)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()
    print(metrics.calinski_harabasz_score(X, y_pred))

if __name__ =='__main__':
    svm()