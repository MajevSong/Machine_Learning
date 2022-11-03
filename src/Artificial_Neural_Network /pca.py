#
# ? PCA - Principal Component Analysis
# Boyut indirgeme yöntemi

# Kullanım Alanları:
# - Gürültü Filitreleme
# - Görselleştirme
# - Öznitelik Çıkarımı
# - Öznitelik Eleme / Dönüştürme
# - Borsa Analizi
# - Sağlık Verileri / Genetik Veriler

# Biz ne için kullanılıyoruz ?
# - Boyut dönüştürme
# - Boyut indirgeme (gereksiz boyutlardan kurtulma veya birleştirme)
# - Değişkenler arasındaki bağlantıları açığa çıkarma

# PCA'nın amacı öyle bir boyut elde edelim ki verileri maksimum ayırt etmek için kullanalım.
# Yeni bir boyut elde ederken bazı veriler kaybolabiliyor. Buna dikkat etmek lazım.
# Eigen Value (Öz Değer) ve Eigen Vector (Öz Yöney)
# - Rastgele bir matris alınır
# - Bu matrisi tek boyutlu bir matris ile çarparız
# - Çarpım şayet çarpanın herhangi bir skalar katını veriyorsa, bu skalar öz değer, bu vektör öz yöneydir.

# PCA Algoritması
# - İndirgenmek istenen boyut k olsun
# - Veriyi Standartlaştır
# - Covariance (Kovaryans) veya Corellation (Korelasyon) matrisinden öz değerleri ve öz vektörleri elde et. Veya SVD kullan
# - Öz değerleri büyükten küçüğe sırala ve k tanesini al
# - Seçilen k özdeğerden W projeksiyon matrisini oluştur
# - Orjinal veri kümesi X'i W kullanarak dönüştür ve k-boyutlu Y uzayını elde et.

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
from sklearn.svm import SVC 

# YSA
from keras.models import Sequential
from keras.layers import Dense

# PCA
from sklearn.decomposition import PCA

veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/wine.csv"
)

X = veriler.iloc[:, 0:13].values
Y = veriler.iloc[:, 13].values

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# pca dönüşümünden önce gelen LR
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

# actual / PCA olmadan çıkan sonuç
print("gercek / PCAsiz")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# actual / PCA sonrası çıkan sonuç
print("gercek / pca ile")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

# PCA sonrası / PCA öncesi
print("pcasiz ve pcali")
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)

# GridSearch

svc=SVC()
svc_pca = SVC()
svc_lda=SVC()
svc_lda.fit(X_train2,y_train)

p=[{'C':[1,2,3,4,5],'kernel':['linear']},
   {'C':[1,2,3,4,5],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]},
   {'C':[1,2,3,4,5],'kernel':['poly'],'degree':[1,2,3,4,5],'gamma':[1,0.5,0.1,0.01,0.001]}]

gs = ms.GridSearchCV(estimator=svc, param_grid=p,scoring='accuracy',cv=4)
gs_search_pca=gs.fit(X_train2,y_train)

en_iyi_parametre = gs_search_pca.best_params_
en_iyi_score = gs_search_pca.best_score_

print(en_iyi_parametre)
print(en_iyi_score)