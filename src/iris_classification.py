import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


veriler = pd.read_csv("../datas/iris.csv")

x = veriler.iloc[:, 1:4].values  # bağımsız değişkenler
y = veriler.iloc[:, 4:].values  # bağımlı değişken
print(y)

# verilerin egitim ve test icin bolunmesi

x_train, x_test, y_train, y_test = train_test_split(
    x, y.ravel(), test_size=0.33, random_state=0
)

# verilerin olceklenmesi

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred_logr = logr.predict(X_test)

print("Logistic Regression")
cm = confusion_matrix(y_test, y_pred_logr)
print(cm)

dogruluk_degeri_logr = metrics.accuracy_score(y_test, y_pred_logr)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_logr))

print("logistic temel model dogruluk degeri = {}".format(dogruluk_degeri_logr))

# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred_knn)
print("KNN Classifier")
print(cm)

dogruluk_degeri_knn = metrics.accuracy_score(y_test, y_pred_knn)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_logr))

print("knn temel model dogruluk degeri = {}".format(dogruluk_degeri_knn))


# Support Vector Machines Classifier
svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print("SVC")
print(cm)

dogruluk_degeri_svc = metrics.accuracy_score(y_test, y_pred_svc)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_svc))

print("svc temel model dogruluk degeri = {}".format(dogruluk_degeri_svc))


# Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred_gnb)
print("GNB")
print(cm)

dogruluk_degeri_gnb = metrics.accuracy_score(y_test, y_pred_gnb)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_gnb))

print("gnb temel model dogruluk degeri = {}".format(dogruluk_degeri_gnb))


# Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_dtc)
print("DTC")
print(cm)

dogruluk_degeri_dtc = metrics.accuracy_score(y_test, y_pred_dtc)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_dtc))

print("dtc temel model dogruluk degeri = {}".format(dogruluk_degeri_dtc))


# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rfc)
print("RFC")
print(cm)

dogruluk_degeri_rfc = metrics.accuracy_score(y_test, y_pred_rfc)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_rfc))

print("rfc temel model dogruluk degeri = {}".format(dogruluk_degeri_rfc))
