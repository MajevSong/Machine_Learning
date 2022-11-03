import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyECLAT import ECLAT


veriler = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/sepet.csv",
    header=None,
)

t = []
for i in range(0, 7501):
    t.append([str(veriler.values[i, j]) for j in range(0, 20)])

# rules = apriori(t, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
# print(list(rules))

# confidence değeri arttıkça ilişki artıyor diyebiliriz.
eclat = ECLAT(data=veriler, verbose=True)
get_ECLAT_index, get_ECLAT_supports = eclat.fit(
    min_support=0.01, min_combination=1, max_combination=3, separator="&", verbose=True
)

help(eclat.fit)
help(eclat.fit_all)
help(eclat.support)

print(get_ECLAT_supports)
