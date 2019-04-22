from sklearn.datasets import fetch_mldata
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib

#mnist = fetch_mldata('MNIST original')   # 国内网络用不了

mnist  = datasets.fetch_mldata("MNIST original", data_home="./")
'''
DESCR键， 描述数据集
·data键， 包含一个数组， 每个实例为一行， 每个特征为一列
·target键， 包含一个带有标记的数组
'''
#print(minst)

X, y = mnist["data"], mnist["target"]
# print(X.shape, y.shape)
'''
(70000, 784)   (70000,)  7万张图片， 每张图片有784个特征 
因为图片是28×28像素， 每个特征代表了一个像素点的强度， 
从0（白色） 到255（黑色）
'''


some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
'''
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
'''
#print(y[36000])  #5.0


# 测试集
X_train, X_test, y_train, y_test = X[ : 60000], X[60000 : ], y[ : 60000], y[60000 : ]

'''
有些机器学习算法对训练实例的顺序敏感， 如果连续输入许多相似的实
例， 可能导致执行性能不佳。 给数据集洗牌正是为了确保这种情况不会发生：
'''
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 训练一个二元分类器
#5和非5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#随机梯度下降（SGD）分类器，使用Scikit-Learn的SGDClassifier类即可。这个分类器的优势是，能够有效处理非常大型的数据集。
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)  #如果你希望得到可复现的结果，需要设置参数random_state。
sgd_clf.fit(X_train, y_train_5)
predict_5 = sgd_clf.predict([some_digit])
#print(predict_5)   # [ true]

# 性能考核
# 使用交叉验证测量精度
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))   #0.9495  0.91815  0.9644

'''
用cross_val_score（） 函数来评估SGDClassifier模型， 采
用K-fold交叉验证法， 3个折叠。 记住， K-fold交叉验证的意思是将训
练集分解成K个折叠（在本例中， 为3折） ， 然后每次留其中1个折叠
进行预测， 剩余的折叠用来训练'''
from sklearn.model_selection import cross_val_score
cross_val_score_5 = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(cross_val_score_5)    # [ 0.9388   0.9642   0.96065]

'''
所有折叠交叉验证的准确率（正确预测的比率）超过95%？看起
来挺神奇的，是吗？不过在你开始激动之前，我们来看一个蠢笨的分
类器，它将每张图都分类成“非5”
'''
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')) # [ 0.912    0.90905  0.9079 ]
'''
没错，准确率超过90%！这是因为只有大约10%的图像是数字
5，所以如果你猜一张图不是5， 90%的时间你都是正确的，简直超越了大预言家！
这说明准确率通常无法成为分类器的首要性能指标，特别是当你
处理偏斜数据集（skewed dataset）的时候（即某些类比其他类更为频
繁）'''

# 混淆矩阵
'''
评估分类器性能的更好方法是混淆矩阵。总体思路就是统计A类
别实例被分成为B类别的次数。例如，要想知道分类器将数字3和数
字5混淆多少次，只需要通过混淆矩阵的第5行第3列来查看。

要计算混淆矩阵，需要先有一组预测才能将其与实际目标进行比
较。当然可以通过测试集来进行预测，但是现在先不要动它（测试集
最好留到项目最后，准备启动分类器时再使用）。作为替代，可以使
用cross_val_predict（）函数
'''
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
'''
与cross_val_score（） 函数一样， cross_val_predict（） 函数同样
执行K-fold交叉验证， 但返回的不是评估分数， 而是每个折叠的预
测。 这意味着对于每个实例都可以得到一个干净的预测（“干净”的意
思是模型预测时使用的数据， 在其训练期间从未见过） 。
现在， 可以使用confusion_matrix（） 函数来获取混淆矩阵了。 只
需要给出目标类别（y_train_5） 和预测类别（y_train_pred） 即
'''
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred))   # [[49651  4928] [  559  4862]]
'''
混淆矩阵中的行表示实际类别， 列表示预测类别。 
本例中第一行表示所有“非5”（负类） 的图片中： 
49651张被正确地分为“非5”类别（真负类） ， 
4928张被错误地分类成了“5”（假正类） ； 
第二行表示所有“5”（正类） 的图片中： 
559张被错误地分为“非5”类别（假负类） ，
4862张被正确地分在了“5”这一类别（真正类） 。 '''

#Scikit-Learn提供了计算多种分类器指标的函数， 精度和召回率也是其一
from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))

#F1分数
from sklearn.metrics import f1_score
print(f1_score(y_train_5, y_train_pred))

# 决策阈值和精度/召回率权衡
y_scores = sgd_clf.decision_function([some_digit])
print("y_scores", y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)
threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

'''
如何决定使用什么阈值呢？
 首先， 使用cross_val_predict（） 函数获取训练集中所有实例的分数， 
 但是这次需要它返回的是决策分数而不是预测结果：'''
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
from sklearn.metrics import precision_recall_curve

# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)   #有错误
'''
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()'''



