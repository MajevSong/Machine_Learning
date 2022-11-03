# Aşağıda, Python ile şimdiye kadar yazdığımız tahmin algoritmalarının
# şablonunu bulabilirsiniz:

# Kütüphaneler

# 1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# numpy ve pandas kütüphanelerini veriyi işlemek ve hafızada yönlendirmek
# için kullanıyoruz (data frame gibi sınıflar için)

# Veri Yükleme

# Veri yükleme aşamasında, verinin yükleneceği dosyanın yanında, bağımlı ve
# bağımsız değişkenleri içeren iki ayrı kolon oluşturulmalıdır. Bağımsız
# değişkenlerin tamamının x isminde bir dataframe içerisinde ve tek bir
# kolondan oluşan bağımlı değişkenleri ise y ismine bir data firma içerisinde
# duracağı kabulü yapılmıştır. Ayrıca X ve Y değişkenleri de numpy dizisi
# olarak bu dataframe'lerden .values özelliğini alır.

veriler = pd.read_csv("/home/valanis/Desktop/Python/Machine_Learning/datas/voice.csv")
x = veriler.iloc[:, 0:19]
x_drop = x.drop(columns=["kurt"])
y = veriler.iloc[:, -1:]
X = x_drop.values
Y = y.values


# LabelEncode islemi anlamli bir degere cevirmek icin kullanilir. Sayisal veriye ceviriyoruz

le = preprocessing.LabelEncoder()
Y[:, 0] = le.fit_transform(veriler.iloc[:, -1:])

# 8. ve 18. sutunlar cikarildi, p value degerleri dusuktu ve onlari cikardiktan sonra R2 degerinde artis gorundu
independent_variables = pd.DataFrame(
    data=X,
    columns=[
        "meanfreq",
        "sd",
        "median",
        "Q25",
        "Q75",
        "IQR",
        "skew",
        "sp.ent",
        "sfm",
        "mode",
        "centroid",
        "meanfun",
        "minfun",
        "maxfun",
        "meandom",
        "mindom",
        "maxdom",
        "dfrange",
    ],
)

dependent_variable = pd.DataFrame(data=Y, columns=["label"])


last_data = pd.concat([independent_variables, dependent_variable], axis=1)
# Korelasyon Matrisi

# Veriler üzerinde karar verirken, kullanılacak önemli ön işleme aşamalarından
# birsi de korelasyon matrisidir ve bu matris ile kolonların birbiri ile olan
# ilişkisi görülebilir.

correlation = last_data.corr()
print(last_data.corr())


# Linear Regression

# Sci-Kit Learn kütüphanesinin genel bir özelliği, fit() fonksiyonu ile eğitmesi
# ve predict fonksiyonu ile tahminde bulunmasıdır. Buradaki örnekte doğrusal
# regresyon (linear regression) üzerinden X ve Y dizileri verilerek bir makine
# öğrenmesi algoritması eğitilmiş ve oluşan model daha sonra OLS ve r2_score
# fonksiyonları ile ölçülmüştür. Ölçüm sırasında gerçek verileri tutan
# Y değişkeni (numpy dizisi) ile lin_reg objesinin predict fonksiyonun döndürdüğü
# tahmin değerleri (yani modelin tahmin etttiği değerler) karşılaştırılmış
# dolayısıyla modelin ne kadar başarılı tahmin yaptığı ölçülmüştür.

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary())
print("Linear R2 degeri:")
print(r2_score(Y, lin_reg.predict((X))))

# Polynomial Regression

# Polynomial regression yöntemi aslında doğrusal regresyondan farklı değildir.
# Hatta aynı nesne ve fonksiyonların kullanıldığı söylenebilir. Buradaki hile,
# verilerin doğrusal regresyona verilmeden önce bir polinom öznitelik fonksiyonuna
# verilmesidir. bu işlem yukarıdaki kodda da gösterildiği üzere, PolynomialFeattures
# nesnesi üzerinden yapılmaktadır.

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)), X)
print(model2.fit().summary())
print("Polynomial R2 degeri:")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


# Destek Vektör Regresyonu ve Ölçekleme (Support Vector Regression , Scaling)

# Destek vektör regresyonunun en önemli özelliği, marjinal verilere karşı
# hassas olmasıdır. Bu yüzden ve verilerin daha iyi tahminini sağlamak için,
# öncelikle standartlaştırma yapılması gerekir. Yukarıdaki kodun ilk parçasında,
# StandardScaler sınıfından türettilmiş sc1 ve sc2 nesneleri sayesinde hem X
# hem de Y dizileri ölçeklenmektedir. Ardından SVR sınıfından rbf çekirdeği (kernel)
# ile üretilen svr_reg isimli nesne ile tahmin modeli oluşturulmakta, bunun için de
# sklearn klasiği olan fit metodu kullanılmaktadır.

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)
from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli, y_olcekli)
model3 = sm.OLS(svr_reg.predict(x_olcekli), x_olcekli)
print(model3.fit().summary())
print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


# Karar Ağacı ile Tahmin (Decision Tree)

# Yapı olarak buraya kadar kullanılan sınıflandırma algoritmalarından
# pek de farklı olmayan karar ağacı sınıflandırması, DecisionTreeRegressor
# sınıfından türetilmiş ve yine X ve Y dizileri üzerinde fit() metodu ile
# bir model inşası için kullanılmıştır.

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)
print("dt ols")
model4 = sm.OLS(r_dt.predict(X), X)
print(model4.fit().summary())
print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)))


# Rassal Orman (Random Forest)

# Alt yapısında karar ağacı kullanan rassal ormanlar (random forest),
# şimdiye kadar elde ettiğimiz yapıya çok benzer şekilde,
# sklearn kütüphanesi içerisinden bir sınıf olarak RandomForestRegressor
# eklemiş (import) ve bu sınıftan da ürettiğimiz nesne ile fit() ve predict()
# metotlarını kullanarak makine öğrenmesini gerçekleştirmiştir.

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())
print("dt ols")
model5 = sm.OLS(rf_reg.predict(X), X)
print(model5.fit().summary())
print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)))
