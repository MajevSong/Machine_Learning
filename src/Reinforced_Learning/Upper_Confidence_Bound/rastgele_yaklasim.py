import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, math

veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/Ads_CTR_Optimisation.csv"
)

# Burada herhangi bir algoritma kullanmadık
# Reklamları rastgele seçtik ve her seferinde farklı bir sonuç döndürdü
# Yaklaşık %10 gibi bir doğruluk payı var. Zaten 10 sütun olduğu için de bu normal bir sonuç

# Random Selection (Rastgele Seçim)
N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0, N):
    name = random.randrange(d)
    secilenler.append(name)
    odul = veriler.values[n, name]  # verilerdeki n.satır = 1 ise ödül 1
    toplam = toplam + odul

plt.hist(secilenler)
plt.show()

# Upper Confidence Bound (Üst güven sınırı)
# Tıklama oranına göre hangi sütun daha çok ödül döndürüyor bunu hesaplayacağız
# Yani en yüksek tahmini ödüle sahip eylemi seçmek için hareket eder.
# * 3 Adımda işlenir:
# 1. Her turda (tur sayısı n olsun), her reklam alternatifi (i için) aşağıdaki sayılar tutulur
# - Ni(n): i sayılı reklamın o ana kadarki tıklama sayısı
# - Ri(n): o ana kadar ki i reklamından gelen toplam ödül
# 2. Yukarıdaki bu iki sayıdan, aşağıdaki değerler hesaplanır.
# - O ana kadar ki her reklamın ortalama ödülü Ri(n)/Ni(n)
# - Güven aralığı için aşağı ve yukarı oynama potansiyeli di(n)
# 3. En yüksek UCB değerine sahip olanı alırız.

N = 10000  # 10.000 tıklama
d = 10  # toplam 10 ilan var
rewards = [0] * d  # ilk başta bütün ilanları ödülü 0
# Ri(n)
reward = [0] * d  # 0'lardan oluşan 10 elemanlı bir liste olacak
# Ni(n)
clicked_advertisement = [0] * d  # o ana kadar ki tıklamalar
result = 0  # toplam ödül
selected = [0]

for n in range(1, N):
    name = 0  # Seçilen ilan
    max_ucb = 0
    for i in range(0, d):
        if clicked_advertisement[i] > 0:
            means = rewards[i] / clicked_advertisement[i]
            delta = math.sqrt(3 / 2 * math.log(n) / clicked_advertisement[i])
            ucb = means + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:
            max_ucb = ucb
            name = i

    selected.append(name)
    clicked_advertisement[name] = clicked_advertisement[name] + 1
    reward = veriler.values[n, name]  # verilerdeki n.satır = 1 ise ödül 1
    rewards[name] = rewards[name] + reward
    result = result + reward

print("Toplam ödül: ", result)

plt.hist(selected)
plt.show()
