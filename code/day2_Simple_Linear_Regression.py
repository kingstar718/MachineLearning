import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 第一步：数据预处理
dataset = pd.read_csv("D:\Git\MachineLearning\data\studentscores.csv")
X = dataset.iloc[ : , : 1].values
Y = dataset.iloc[ : , 1 ].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=1/4, random_state=0)

# 第二步：训练集使用简单线性回归模型来训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 第三步：预测结果
Y_pred = regressor.predict(X_test)

# 第四步：可视化
# 训练集结果可视化

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()

# 测试集结果可视化

plt.scatter(X_test , Y_test, color = 'pink')
plt.plot(X_test , regressor.predict(X_test), color ='yellow')
plt.show()
