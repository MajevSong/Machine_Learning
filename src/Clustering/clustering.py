# Created by M.S

# ! Clustering and Segmentation

# * Gözetimli Öğrenme;
# Bağımlı ve bağımsız değişkenler, yani girdi ve çıktılar bir arada olur.
# Gözetimli öğrenmede, makineyi "etiketlenmiş" verileri kullanarak eğitirsiniz.

# * Gözetimsiz Öğrenme;
# Bu öğrenme biçiminde çıktılar çalışmanın içinde bulunmaz. Gözlemlenen birimler
# benzer özelliklerine göre bir araya getirilir. Gözetimsiz öğrenme, etiketlenmemiş
# veri kümelerini analiz etmek ve kümelemek için yapay öğrenme algoritmalarını kullanır.
# Bu algoritmalar, insan müdahelesine ihtiyaç duymadan verileri sınıflandırırlar.
# Gözetimsiz öğrenme modelleri 3 ana göre için kullanılır;
# ? Kümeleme, İlişkilendirme ve Boyutsallık Azaltma

# * Kümeleme:
# Etiketlenmemiş verileri benzerliklerine veya farklılıklarına göre gruplandırmak
# için kullanılan bir veri madenciliği tekniğidir. Örnek: K-MEANS

# * İlişkilendirme:
# Belirli bir veri kümesindeki değişkenler arasındaki ilişkileri bulmak için farklı
# kurallar kullanan başka bir gözetimsiz öğrenme yöntemidir. Bu yöntemler, pazar sepeti
# analizi ve öneri motorlarında “bu ürünü satın alan müşteriler şunları da aldı”
# önerilerini oluşturmak için sıklıkla kullanılmaktadır.

# * Boyut azaltma:
# Belirli bir veri kümesindeki özelliklerin (veya boyutların) sayısı çok yüksek olduğunda
# kullanılan bir öğrenme tekniğidir. Veri bütünlüğünü korurken, veri girişlerinin sayısını
# yönetilebilir bir boyuta indirir.

# ? Kullandığı alanlar

# A. Müşteri Segmentation
# B. Pazar Segmentation
# C. Bilgisayar ile Görü

# ? Algoritmalar

# * K-MEANS algoritması
# Kaç küme olacağı kullanıcıdan parametre olarak seçilir
# Rastgele olarak k merkez noktası seçilir
# Her veri örneği en yakın merkez notkasına göre ilgili kümeye atanır
# Her küme için yeni merkez noktaları hesaplanarak merkez noktaları kaydırılır
# Yeni merkez noktalarına göre

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/datas/musteriler.csv"
)

X = veriler.iloc[:, 3:].values

kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans_fit = kmeans.fit(X)

print(kmeans_fit.cluster_centers_)
sonuclar = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1, 11), sonuclar)
plt.show()

kmeans = KMeans(n_clusters=4, init="k-means++", random_state=123)
kmeans_fit_predict = kmeans.fit_predict(X)

plt.scatter(
    X[kmeans_fit_predict == 0, 0], X[kmeans_fit_predict == 0, 1], s=100, c="red"
)
plt.scatter(
    X[kmeans_fit_predict == 1, 0], X[kmeans_fit_predict == 1, 1], s=100, c="blue"
)
plt.scatter(
    X[kmeans_fit_predict == 2, 0], X[kmeans_fit_predict == 2, 1], s=100, c="green"
)
plt.scatter(
    X[kmeans_fit_predict == 3, 0], X[kmeans_fit_predict == 3, 1], s=100, c="yellow"
)

plt.title("KMeans")
plt.show()

# * Hiyerarşik Bölütleme(Kümeleme) Algoritması
# Agglomerative ve Divisive olarak ikiye ayrılır.

# Agglomerative -> Aşağıdan yukarıya yaklaşım
# Divisive -> Yukarıdan aşağıya yaklaşım

# Her veri tek bir küme / bölüt ile başlar
# En yakın ikişer komşuyu alıp ikişerli küme/bölüt oluşturulur
# En yakın iki kümeyi alıp yeni bir bölüt oluşturulur
# Bir önceki adım, tek bir bölüt/küme olana kadar devam eder.

# Mesafe ölçümü için euclidean kullanacagız. küme sayısını 3 yaptık.
agglomerative_clustering = AgglomerativeClustering(
    n_clusters=4, affinity="euclidean", linkage="ward"
)
#  hem eğitip hem de tahmin ediyoruz
y_pred_agg = agglomerative_clustering.fit_predict(X)

print(y_pred_agg)

plt.scatter(X[y_pred_agg == 0, 0], X[y_pred_agg == 0, 1], s=100, c="red")
plt.scatter(X[y_pred_agg == 1, 0], X[y_pred_agg == 1, 1], s=100, c="blue")
plt.scatter(X[y_pred_agg == 2, 0], X[y_pred_agg == 2, 1], s=100, c="green")
plt.scatter(X[y_pred_agg == 3, 0], X[y_pred_agg == 3, 1], s=100, c="yellow")
plt.title("Hierarchical Clustering")
plt.show()

# Dendrogram
# En uzun çizginin oldugu yer bizim almak istedigimiz kesişim yeri olacak.
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.show()

# Grafikte en mantıklı alınması gerekilen yer 2'dir. 4 de alınabilir.
