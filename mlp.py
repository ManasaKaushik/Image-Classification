import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.svm import SVC
from statistics import mean

X = pd.read_csv("data/X.csv",sep = ' ', header=None,dtype=float)
X = X.values

Y = pd.read_csv("data/y_bush_vs_others.csv",sep = ' ', header=None,dtype=float)
Bush = Y.values.ravel()

Y = pd.read_csv("data/y_williams_vs_others.csv",sep = ' ', header=None,dtype=float)
Williams = Y.values.ravel()

X.shape

np.sum(Bush)
np.sum(Williams)

gamma_range = [1e-7,1e-6,1e-5,1e-2,1e-1,1]
svc_results = []

svc_poly_mean_williams = []
gamma_range = [0.00001,0.0001,0.0005,0.001,0.005,0.01]
C_params = [100,120,150,200,250,300,1000]
print("==================== C values, Kernel = Poly, Degree =2, Bush  ==================")
for C in C_params:
    print(" +++++++++++++++++  C = " + str(C) + "+++++++++++++++++++++++")
    svc = SVC(C=C,kernel='poly',degree=2)
    stratified_cv_results = cross_validate(svc,X,Bush, cv=StratifiedKFold(n_splits = 3,shuffle = True,random_state = 7459),
                                           scoring=('precision','recall','f1'),return_train_score=False,n_jobs=-1)     
    print("==================" + str(C) + "\t" + "==================")
    print(stratified_cv_results)
    svc_poly_mean_williams.append(stratified_cv_results)
    print("\n")


svct = SVC(C=0.001, kernel='linear')
stratified_cv_results = cross_validate(svct,X,Bush, cv=StratifiedKFold(n_splits = 3,shuffle = True,random_state = 7459),
                                           scoring=('precision','recall','f1'),return_train_score=False,n_jobs=-1)




Y = pd.read_csv("data/y_bush_vs_others.csv",sep = ' ', header=None,dtype=float)
Bush = Y.values.ravel()

Z = pd.read_csv("data/y_williams_vs_others.csv",sep = ' ', header=None,dtype=float)
Williams = Z.values.ravel()

pca = PCA(n_components=0.5, copy=True, random_state=7459)


XT = pca.fit(X).transform(X)


BT = pca.fit(Bush).transform(Bush)

pknn = KNeighborsClassifier(1)
stratified_cv_results = cross_validate(pknn, XT, BT, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 7459), scoring=('precision', 'recall', 'f1'), return_train_score=False, n_jobs=-1)



