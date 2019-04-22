import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from  sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# 线性分类SVM
def first_code_example():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
    # print(X, y)
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])
    svm_clf.fit(X, y)
    print(svm_clf.predict([[5.5, 1.7]]))

# 数据展示函数
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

# 数据展示
def non_linear_dataset():
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


# 非线性SVM
def non_linear_svm():
    from sklearn.datasets import make_moons
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures

    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])
    polynomial_svm_clf.fit(X, y)
    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.savefig("moons_polynomial_svc_plot")
    plt.show()

#  不同核的SVM分类器
'''
    寻找正确的超参数值的常用方法是网格搜索（见第2章） 。 先进行一次粗略的网格搜索， 然后在最好的值附近展开一轮更精细的网
    格搜索， 这样通常会快一些。 多了解每个超参数实际上是用来做什么的， 有助于你在超参数空间层正确搜索'''
def diff_degree_svm():
    from sklearn.svm import SVC
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    # d=3 c=1 C = 5
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(X, y)
    # d = 10   c = 100  C = 5
    poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
    poly100_kernel_svm_clf.fit(X, y)

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=3, r=1, C=5$", fontsize=18)

    plt.subplot(122)
    plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=10, r=100, C=5$", fontsize=18)

    plt.savefig("moons_kernelized_polynomial_svc_plot")
    plt.show()

# 不同的高斯RBF核作用于SVC类
def diff_rbf_svm():
    from sklearn.svm import SVC
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

    gamma1, gamma2 = 0.1, 5
    C1, C2 = 0.001, 1000
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

    svm_clfs = []
    for gamma, C in hyperparams:
        rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
        rbf_kernel_svm_clf.fit(X, y)
        svm_clfs.append(rbf_kernel_svm_clf)

    plt.figure(figsize=(11, 7))

    for i, svm_clf in enumerate(svm_clfs):
        plt.subplot(221 + i)
        plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
        plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        gamma, C = hyperparams[i]
        plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

    plt.savefig("moons_rbf_svc_plot")
    plt.show()


if __name__=="__main__":
    # 线性SVM
    #first_code_example()

    # 非线性数据展示
    #non_linear_dataset()

    # 非线性SVM分类器
    #non_linear_svm()

    # 不同内核的非线性SVM
    # diff_degree_svm()

    # 不同的高斯RBF核作用于SVC类
    '''
    增加gamma值会使钟形曲线变得更窄（图5-8的左图） ， 
    因此每个实例的影响范围随之变小：     决策边界变得更不规则， 开始围着单个实例绕弯。 
    反过来， 减小gamma值使钟形曲线变得更宽， 因而每个实例的影响范围增大， 决策边界变得更平坦。
     所以就像是一个正则化的超参数： 模型过度拟合， 就降低它的值， 如果拟合不足则提升它的值（类似超参数C） 。'''
    diff_rbf_svm()

    '''
    有这么多的核函数，该如何决定使用哪一个呢？
    有一个经验法则是，永远先从线性核函数开始尝试（要记住， LinearSVC比SVC（kernel="linear"）快得多），
    特别是训练集非常大或特征非常多的时候。如果训练集不太大，你可以试试高斯RBF核，大多数情况下它都非常好用。
    如果你还有多余的时间和计算能力，你可以使用交叉验证和网格搜索来尝试一些其他的核函数，
    特别是那些专门针对你的数据集数据结构的核函数。'''
    pass