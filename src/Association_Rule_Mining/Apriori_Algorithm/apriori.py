##
# ? Apriori Algoritması (ARM/ARL)
# Birliktelik Kural Çıkarımı
# Tekrar eden eylemlerin yakalanması
# Mesela sigara içenler naneli şeker alıyor.
# Sigara ile naneli şeker arasında nasıl bir ilişki olabilir ? vs.

# * Nerelerde kullanılır:
# Complex Event Processing
# Davranış Tahmini
# Yönlendirilmiş ARM
# Zaman Serisi Analizi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from apyori import apriori

veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/sepet.csv",
    header=None,
)

t = []
for i in range(0, 7501):
    t.append([str(veriler.values[i, j]) for j in range(0, 20)])

rules = apriori(t, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
print(list(rules))
