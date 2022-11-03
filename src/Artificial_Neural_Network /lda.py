# Sınıflar arasındaki farklılığı maximize etmek için kullanılır

import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import model_selection as ms
from sklearn import model_selection as ms
from sklearn.svm import SVC 

# YSA
from keras.models import Sequential
from keras.layers import Dense

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/wine.csv"
)

X = veriler.iloc[:, 0:13].values
Y = veriler.iloc[:, 13].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# LogisticRegression- LDA donusumunden once
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# tahminler
y_pred = classifier.predict(X_test)

# LDA
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# LDA donusumunden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

# LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)

# actual / PCA olmadan çıkan sonuç
print("gercek / LDAsiz")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# actual / PCA sonrası çıkan sonuç
print("gercek / lda ile")
cm2 = confusion_matrix(y_test, y_pred_lda)
print(cm2)

# LDA sonrası / orijinal
print("lda ve orijinal")
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)

svc=SVC()
svc_pca = SVC()
svc_lda=SVC()
svc_lda.fit(X_train_lda,y_train)

p=[{'C':[1,2,3,4,5],'kernel':['linear']},
   {'C':[1,2,3,4,5],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]},
   {'C':[1,2,3,4,5],'kernel':['poly'],'degree':[1,2,3,4,5],'gamma':[1,0.5,0.1,0.01,0.001]}]

gs = ms.GridSearchCV(estimator=svc, param_grid=p,scoring='accuracy',cv=4)
gs_search_lda=gs.fit(X_train_lda,y_train)

en_iyi_parametre = gs_search_lda.best_params_
en_iyi_score = gs_search_lda.best_score_

print(en_iyi_parametre)
print(en_iyi_score)