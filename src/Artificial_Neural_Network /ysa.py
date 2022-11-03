# Artifical Neural Network

# Çıktı eğer yanlış tahmin ederse penalty dedigimiz hata katsayı ağırlıklara eklenerek tekrar hesaplanır.
# Öğrenme oranını bulmak önemli.
# Öğrenme oranı yüksek olmak zorunda değildir. Azaltılarak uygulanırsa daha iyi olur.

# Gradient Descient metodu ile optimum öğrenme eğrisini bulmaya çalışırız;
# Ama bazı problemlerle karşılaşırız bunları gidermek için bazı yöntemler kullanabiliriz.
# Stokastik, Mini Batch ve Batch yaklaşımı ile verilerin alçaltılıp alçaltmayacağını belirtebiliriz.
# Stokastik yöntemi ile veriyi tek tek değerlendirip ona göre bir penalty belirtip öğrenme eğrisini bulmaya çalışır.
# Minibatch yöntemi ile veriyi parça parça inceleyerek ona göre alçaltma yapılır veya yapılmaz.
# Batch yaklaşımı ise tüm veriyi değerlendirip ona göre bir alçaltma yapar veya yapmaz. Bunu her seferinde tekrarlar.

# Bir yapay sinir ağı girişten çıkışa doğru yada tam tersi şekilde öğrenim gerçekleştirebilir


# ? Forward Propagation (İleri Yayılım)
# Neural Network’te input’tan başlayarak output’a giden yolculuğumuz ileri yön (forward) olarak isimlendirilmektedir.
# Her node’a giren weights, eldeki değer ile (input ise x feature değeri ile, hidden layer ise o node’a giren önceki çarpımların toplamından gelen değer ile) çarpılır ve bias eklenir.

# ? Backward Propagation (Geriye Yayılım) alogirtma adımları
# Bu algoritma hataları output’tan input’a doğru azaltmaya çalışmasından dolayı geri yayılım ismini almıştır. Gradyan iniş olarak adlandırılan bir teknik kullanarak ağırlık alanındaki hata fonksiyonunun minimum değerini arar.

# Bu yöntem, ağırlık ve bias değerlerini değiştirerek hatayı azaltmaya çalışır.

# ! Algoritmanın adımları;

# 1. Bütün ağı rasgele sayılarla (sıfıra yakın ama sıfırdan farklı) ilkendir.
# 2. Veri kümesinden ilk satır (her öznitelik bir nöron olacak şekilde) giriş katmanından verilir
# 3. İleri yönlü yayılım yapılarak, YSA istenen sonucu verene kadar güncellenir.
# 4. Gerçek ve çıktı arasındaki fark alınarak hata(error) hesaplanır.
# 5. Geri yayılım yapılarak, her sinapsis üzerindeki ağırlık, hatadan sorumlu olduğu miktarda değiştirilir.
# Değiştirilme miktarı ayrıca öğrenme oranına da bağlıdır.
# 6. Adım 1-5 arasındaki adımları istenen sonucu elde edene kadar güncelle. (Takviyeli öğrenme(Reinforced Learning))
# veya eldeki bütün verileri ilgili ağda çalıştırdıktan sonra bir seferde güncelleme yap. (Yığın öğrenme(batch learning))
# 7. Bütün eğitim kümesi çalıştırıldıktan sonra bir çağ/tur(epoch) tamamlanmış olur. Aynı veri kümeleri kullanılarak cağ/tur tekrarları yapılır.
# epoch = veri üzerinde kaç tur atacağıdır
# learning_rate = öğrenme oranıdır.

# Keras ile bir yapay sinir ağı oluşturulur
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# YSA
from keras.models import Sequential
from keras.layers import Dense

veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/Churn_Modelling.csv"
)

X = veriler.iloc[:, 3:13].values
Y = veriler.iloc[:, 13].values


le = preprocessing.LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

le2 = preprocessing.LabelEncoder()
X[:, 2] = le2.fit_transform(X[:, 2])

ohe = ColumnTransformer(
    [("ohe", OneHotEncoder(dtype=float), [1])], remainder="passthrough"
)
X = ohe.fit_transform(X)
X = X[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0
)


sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Yapay Sinir Ağı

# Bir yapay sinir ağı oluşturuyoruz
classifier = Sequential()

# Neuron'lar ekleniyor
# Hidden layer
classifier.add(Dense(6, activation="relu", input_dim=11))

classifier.add(Dense(6, activation="relu"))

classifier.add(Dense(1, activation="sigmoid"))

# binomial değerler için loss = binary_crossentropy kullanılır.
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(X_train, y_train, epochs=50)

y_pred = classifier.predict(X_test)

y_pred = y_pred > 0.5

cm = confusion_matrix(y_test, y_pred)

print(cm)

