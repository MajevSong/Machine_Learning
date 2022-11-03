#
# ? Thompson Samples
# Algorithm steps:
# Adım 1: her aksiyon için aşağıdaki iki sayıyı hesaplayınız
# - Ni1(n): o ana kadar ödül olarak 1 gelme sayısı
# - Ni0(n): o ana kadar ödül olarak 0 gelme sayısı
# Adım 2: Her ilan için aşağıdaki verilen Beta dağılımında bir rasgele sayı üretiyoruz
# Adım 3: En yüksek beta değerine sahip olanı seçiyoruz.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/Ads_CTR_Optimisation.csv"
)

N = 10000  # 10.000 tıklama
d = 10  # toplam 10 ilan var
# Ri(n)
reward = [0] * d  # 0'lardan oluşan 10 elemanlı bir liste olacak
result = 0  # toplam ödül
selected = [0]
ones = [0] * d
zeros = [0] * d

for n in range(1, N):
    name = 0  # Seçilen ilan
    max_th = 0
    for i in range(0, d):
        random_beta = random.betavariate(ones[i] + 1, zeros[i] + 1)
        if random_beta > max_th:
            max_th = random_beta
            name = i

    selected.append(name)
    reward = veriler.values[n, name]  # verilerdeki n.satır = 1 ise ödül 1
    if reward == 1:
        ones[name] = ones[name] + 1
    else:
        zeros[name] = zeros[name] + 1

    result = result + reward

print("Toplam ödül: ", result)

plt.hist(selected)
plt.show()
