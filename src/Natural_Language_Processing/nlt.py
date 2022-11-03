#
# ? Natural Language Processing
# Doğal Dil İşleme

import pandas as pd
import numpy as np
import re

yorumlar = pd.read_csv(
    "/home/valanis/Desktop/Python/Machine_Learning/Teachcareer/Project/datas/Restaurant_Reviews.csv"
)

yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Review"][0])
yorum = yorum.lower()
yorum = yorum.split()
