# 第1步：导入库
import numpy as np
import pandas as pd


# 第2步：导入数据集
dataset = pd.read_csv('D:\Git\MachineLearning\data\Data.csv')
X = dataset.iloc[:, : -1].values #.iloc[行，列]
Y = dataset.iloc[:, 3].values  #: 全部行 or 列；[a]第a行 or 列    [a,b,c]第 a,b,c 行 or 列
#print(X, Y)


#第3步：处理丢失数据   用整列的平均值或中间值替换丢失的数据
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

# 第4步：解析分类数据   将不是数字值的标签值解析成数字
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#创建虚拟变量
onehotencoder  = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(X, Y)

# 第5步：拆分数据集为训练集合和测试集合
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#第6步：特征量化   特征标准化/Z值标准化
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

