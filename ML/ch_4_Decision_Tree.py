# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def first_Decision_Tree():
    iris = load_iris()
    X = iris.data[: , 2:]
    y = iris.target
    # print(X, y)
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)
    # 要将决策树可视化，首先，使用export_graphviz（）方法输出一
    # 个图形定义文件，命名为iris_tree.dot：
    '''
    from sklearn.tree import export_graphviz
    export_graphviz(
        tree_clf,
        out_file=image_path("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )'''
    print(tree_clf.predict_proba([[5, 1.5]]))  #Setosa鸢尾花， 0%（0/54） ； Versicolor鸢尾花， 90.7%（49/54） ； Virginica鸢尾花， 9.3%（5/54）
    print(tree_clf.predict([[5, 1.5]])) #输出Versicolor鸢尾花（类别1） ， 因为它的概率最高。

from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

def moons_Decision_Tree():
    from sklearn.datasets import make_moons
    Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

    deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
    deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
    deep_tree_clf1.fit(Xm, ym)
    deep_tree_clf2.fit(Xm, ym)

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
    plt.title("No restrictions", fontsize=16)
    plt.subplot(122)
    plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
    plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)

    #save_fig("min_samples_leaf_plot")
    plt.savefig("min_samples_leaf_plot")
    plt.show()

def first_Regression_trees():
    # Quadratic training set + noise
    np.random.seed(42)
    m = 200
    X = np.random.rand(m, 1)
    y = 4 * (X - 0.5) ** 2
    y = y + np.random.randn(m, 1) / 10

    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg.fit(X, y)
    tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
    tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
    tree_reg1.fit(X, y)
    tree_reg2.fit(X, y)

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_regression_predictions(tree_reg1, X, y)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    plt.text(0.21, 0.65, "Depth=0", fontsize=15)
    plt.text(0.01, 0.2, "Depth=1", fontsize=13)
    plt.text(0.65, 0.8, "Depth=1", fontsize=13)
    plt.legend(loc="upper center", fontsize=18)
    plt.title("max_depth=2", fontsize=14)

    plt.subplot(122)
    plot_regression_predictions(tree_reg2, X, y, ylabel=None)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    for split in (0.0458, 0.1298, 0.2873, 0.9040):
        plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
    plt.text(0.3, 0.5, "Depth=2", fontsize=13)
    plt.title("max_depth=3", fontsize=14)

    #save_fig("tree_regression_plot")
    plt.savefig("tree_regression_plot")
    plt.show()

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

def second_Regression_Tree():
    # Quadratic training set + noise
    np.random.seed(42)
    m = 200
    X = np.random.rand(m, 1)
    y = 4 * (X - 0.5) ** 2
    y = y + np.random.randn(m, 1) / 10

    from sklearn.tree import DecisionTreeRegressor
    tree_reg1 = DecisionTreeRegressor(random_state=42)
    tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)  # s设置
    tree_reg1.fit(X, y)
    tree_reg2.fit(X, y)

    x1 = np.linspace(0, 1, 500).reshape(-1, 1)
    y_pred1 = tree_reg1.predict(x1)
    y_pred2 = tree_reg2.predict(x1)

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", fontsize=18, rotation=0)
    plt.legend(loc="upper center", fontsize=18)
    plt.title("No restrictions", fontsize=14)

    plt.subplot(122)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=18)
    plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)

    #save_fig("tree_regression_regularization_plot")
    plt.savefig("tree_regression_regularization_plot")
    plt.show()

if __name__ == "__main__":
    #first_Decision_Tree()
    '''
    在Scikit-Learn中，这由超参数max_depth控制（默认值为None，意味着无限制）。
    减小max_depth可使模型正则化，从而降低过度拟合的风险。
    DecisionTreeClassifier类还有一些其他的参数，同样可以限制决策树的形状：
     min_samples_split（分裂前节点必须有的最小样本数）， 
     min_samples_leaf（叶节点必须有的最小样本数量），
     min_weight_fraction_leaf（跟min_samples_leaf一样，但表现为加权实例总数的占比）， 
     max_leaf_nodes（最大叶节点数量），
     max_features（分裂每个节点评估的最大特征数量）。    
      增大超参数min_*或是减小max_*将使模型正则化。'''

    # 左图使用默认参数（即无约束）来训练决策树，右图的决策树应
    # 用min_samples_leaf=4进行训练。很明显，左图模型过度拟合，右图
    # 的泛化效果更佳
    #moons_Decision_Tree()

    # 回归树
    #first_Regression_trees()

    # 参数改变
    second_Regression_Tree()
    pass