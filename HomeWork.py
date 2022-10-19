#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:16:41 2022

@author: Mücahit Söylemez
@HomeWork : Arabanın özelliklerine göre fiyatını tahmin etme
@Kullanılan Yöntemler: Linear, Coklu Linear Regression, Polynomial, SVR, Tree regression
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as trs
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing as pr
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

veri = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/HomeWork/car_data.csv")

ypk = veri[['Year', 'Present_Price', 'Kms_Driven']]
transmission = veri.iloc[:, 7:8].values
y = veri[['Selling_Price']]
Y = y.values
X_df = pd.DataFrame(data=ypk.values, columns=[
                    'Year', 'Present_Price', 'Kms_Driven'])

le = pr.LabelEncoder()

transmission[:, 0] = le.fit_transform(veri.iloc[:, 7:8])

ohe = pr.OneHotEncoder()

transmission = ohe.fit_transform(transmission).toarray()
transmission_sonuc = pd.DataFrame(
    data=transmission[:, :1], columns=["Transmission"])

X_df_son = pd.concat([X_df, transmission_sonuc], axis=1)

# 1. Çoklu Linear Regression

# Tum degiskenleri alarak bir model insa etmek her zaman faydalı olmayabilir.
# O yuzden sonuc ile alakası olmayan degiskenleri modelimizden cikarmak daha iyi sonuc vermesini saglayabilir.
# Bunun icin de geriye,ileriye ve her ikisini de kullanacagimiz modeller insa edebiliriz.
# Significant Level (0.05) alınır, P degeri bu degere gore bakilir.
# Backward Elimination => En yuksek P degerine sahip degisken sistemden cikarilir
# Forward Selection => En dusuk P degerine sahip degisken sabit birakilip yeni degiskenler eklenir
# Bidirectional Elimination => Hem Backward hem Forward yontemi kullanilir. Degiskenleri eleyerek ve ekleyerek ilerleriz.

x_train, x_test, y_train, y_test = trs(
    X_df_son, Y, test_size=0.33, random_state=0)

X_Linear = np.append(arr=np.ones((299, 1)).astype(int),
                     values=X_df_son, axis=1)
X_1 = X_df_son.iloc[:, [0, 1, 2, 3]].values
r_ols = sm.OLS(Y, X_1).fit()
print(r_ols.summary())

lr = LinearRegression()
lr.fit(x_train, y_train)

y_coklu_linear_prediction = lr.predict(x_test)

# 2. Polynomial Regression

poly_reg = pr.PolynomialFeatures()
x_poly = poly_reg.fit_transform(X_df_son)

x_train_poly, x_test_poly, y_train, y_test = trs(
    x_poly, Y, test_size=0.33, random_state=0)

lr2 = LinearRegression()
lr2.fit(x_train_poly, y_train)
Y_prediction_poly = lr2.predict(x_test_poly)

# 3. SVR

# StandardScale
sc = pr.StandardScaler()
x_train_olc = sc.fit_transform(x_train)
x_test_olc = sc.fit_transform(x_test)
y_train_olc = sc.fit_transform(y_train)
y_test_olc = sc.fit_transform(y_test)

svr_reg = SVR(kernel="rbf")
# ravel() fonksiyonu matrisimizi düzlestirip 1 boyutlu diziye donusturur.
svr_reg.fit(x_train_olc, y_train_olc.ravel())
y_prediction_svr = svr_reg.predict(x_test_olc)

# 4. Karar Agaci

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x_train, y_train)
y_prediction_dt = dtr.predict(x_test)
