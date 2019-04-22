import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Git\MachineLearning\data\Startups_50.csv')
X = dataset.iloc[ : , : -1].values
Y = dataset.iloc[ : ,  4].values

# 将类别数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[ : , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#躲避虚拟变量陷阱
X = X[:, 1:]

# 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#第6步：特征量化   特征标准化/Z值标准化(原代码没有)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# 在训练集上训练多元线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
