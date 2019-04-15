import pandas as pd
import numpy as np

dataset = pd.read_csv("D:\Git\MachineLearning\data\50_Startups.csv")
X = dataset.iloc[ : , : -1].values
Y = dataset.iloc[ : , 4].values

# 将类别数据数字化