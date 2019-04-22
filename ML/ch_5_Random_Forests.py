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
CHAPTER_ID = "ensembles"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


def Voting_classifiers():
    heads_proba = 0.51
    coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
    cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)
    plt.figure(figsize=(8, 3.5))
    plt.plot(cumulative_heads_ratio)
    plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
    plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
    plt.xlabel("Number of coin tosses")
    plt.ylabel("Heads ratio")
    plt.legend(loc="lower right")
    plt.axis([0, 10000, 0.42, 0.58])
    #save_fig("law_of_large_numbers_plot")
    plt.savefig("law_of_large_numbers_plot")
    plt.show()

def three_classifier():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard')
    voting_clf.fit(X_train, y_train)
    # 每个分类器在测试集上的准确率：
    from sklearn.metrics import accuracy_score
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

    '''
    如果所有分类器都能够估算出类别的概率（即有predict_proba（） 方法） ， 那么你可以将概率在所有单个分类器上平均，
     然后让Scikit-Learn给出平均概率最高的类别作为预测。 这被称为软投票法。 
     通常来说， 它比硬投票法的表现更优， 因为它给予那些高度自信的投票更高的权重。 
     而所有你需要做的就是用voting="soft"代替voting="hard"， 并确保所有分类器都可以估算出概率。 
     默认情况下， SVC类是不行的， 所以你需要将其超参数probability设置为True（这会导致SVC使用交叉验证来估算类别概率， 
     减慢训练速度，并会添加predict_proba（） 方法） 。 
     如果修改上面代码为使用软投票， 你会发现投票分类器的准确率达到91%以上'''
    log_clf1 = LogisticRegression(random_state=42)
    rnd_clf1 = RandomForestClassifier(random_state=42)
    svm_clf1 = SVC(probability=True, random_state=42)

    voting_clf1 = VotingClassifier(
        estimators=[('lr', log_clf1), ('rf', rnd_clf1), ('svc', svm_clf1)],
        voting='soft')
    voting_clf1.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    for clf1 in (log_clf1, rnd_clf1, svm_clf1, voting_clf1):
        clf1.fit(X_train, y_train)
        y_pred = clf1.predict(X_test)
        print(clf1.__class__.__name__, accuracy_score(y_test, y_pred))


'''
另一种方法是每个预测器使用的算法相同，但是在不同的训练集随机子集上进行训练。
采样时如果将样本放回，这种方法叫作bagging（bootstrap aggregating[2]的缩写，也叫自举汇聚法）；
采样时样本不放回，这种方法则叫用pasting。
换句话说， bagging和pasting都允许训练实例在多个预测器中被多次采样，但是只有bagging允许训练实例被同一个预测器多次采样。'''

def bagging_ensembles():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    bag_clf = BaggingClassifier(                # 由500个决策树组成的bagging集成
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))

    tree_clf = DecisionTreeClassifier(random_state=42)   #一个是单个的决策树
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred_tree))

    # 将设置 oob_score=True，就可以请求在训练结束后自动进行包外评估。
    bag_clf_true = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42, oob_score=True)
    bag_clf_true.fit(X_train, y_train)
    y_pred_true = bag_clf_true.predict(X_test)
    print(accuracy_score(y_test, y_pred_true))

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_decision_boundary(tree_clf, X, y)
    plt.title("Decision Tree", fontsize=14)
    plt.subplot(122)
    plot_decision_boundary(bag_clf, X, y)
    plt.title("Decision Trees with Bagging", fontsize=14)
    # save_fig("decision_tree_without_and_with_bagging_plot")
    plt.savefig("decision_tree_without_and_with_bagging_plot")
    plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

def random_forests():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)

    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))

    from sklearn.ensemble import RandomForestClassifier

    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    rnf_pred = np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions
    print(rnf_pred)


def feature_importance():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rnd_clf.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)
    print(rnd_clf.feature_importances_)

from sklearn.ensemble import RandomForestClassifier
def feature_importance2():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')  # 数据无法读入
    rnd_clf = RandomForestClassifier(random_state=42)
    rnd_clf.fit(mnist["data"], mnist["target"])

    def plot_digit(data):
        image = data.reshape(28, 28)
        plt.imshow(image, cmap=matplotlib.cm.hot, interpolation="nearest")
        plt.axis("off")

    plot_digit(rnd_clf.feature_importances_)

    cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
    cbar.ax.set_yticklabels(['Not important', 'Very important'])

    plt.savefig("mnist_feature_importance_plot")
    plt.show()


from sklearn.tree import DecisionTreeClassifier
def AdaBoost():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from sklearn.ensemble import AdaBoostClassifier
    # 训练了一个AdaBoost分类器，它基于200个单层决策树
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    ada_clf.fit(X_train, y_train)
    plot_decision_boundary(ada_clf, X, y)


from sklearn.svm import SVC

def AdaBoost2():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    m = len(X_train)
    plt.figure(figsize=(11, 4))
    for subplot, learning_rate in ((121, 1), (122, 0.5)):
        sample_weights = np.ones(m)
        for i in range(5):
            plt.subplot(subplot)
            svm_clf = SVC(kernel="rbf", C=0.05, random_state=42)
            svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = svm_clf.predict(X_train)
            sample_weights[y_pred != y_train] *= (1 + learning_rate)
            plot_decision_boundary(svm_clf, X, y, alpha=0.2)
            plt.title("learning_rate = {}".format(learning_rate), fontsize=16)

    plt.subplot(121)
    plt.text(-0.7, -0.65, "1", fontsize=14)
    plt.text(-0.6, -0.10, "2", fontsize=14)
    plt.text(-0.5, 0.10, "3", fontsize=14)
    plt.text(-0.4, 0.55, "4", fontsize=14)
    plt.text(-0.3, 0.90, "5", fontsize=14)
    #save_fig("boosting_plot")
    plt.savefig("boosting_plot")
    plt.show()

def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

# 梯度提升
def Gradient_Boosting():
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

    from sklearn.tree import DecisionTreeRegressor
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X, y)
    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X, y2)
    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X, y3)
    X_new = np.array([[0.8]])
    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    print(y_pred)

    plt.figure(figsize=(11, 11))

    plt.subplot(321)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-",
                     data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Residuals and tree predictions", fontsize=16)

    plt.subplot(322)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$",
                     data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Ensemble predictions", fontsize=16)

    plt.subplot(323)
    plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+",
                     data_label="Residuals")
    plt.ylabel("$y - h_1(x_1)$", fontsize=16)

    plt.subplot(324)
    plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
    plt.ylabel("$y$", fontsize=16, rotation=0)

    plt.subplot(325)
    plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
    plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
    plt.xlabel("$x_1$", fontsize=16)

    plt.subplot(326)
    plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8],
                     label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$y$", fontsize=16, rotation=0)

    #save_fig("gradient_boosting_plot")
    plt.savefig("gradient_boosting_plot")
    plt.show()

#GBRT集成——预测器太少（左图） 和预测器太多（右图）
def lessOrMore():
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

    from sklearn.ensemble import GradientBoostingRegressor
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
    gbrt.fit(X, y)

    gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
    gbrt_slow.fit(X, y)
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
    plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

    plt.subplot(122)
    plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)

    plt.savefig("gbrt_learning_rate_plot")
    plt.show()

from sklearn.ensemble import GradientBoostingRegressor
def Gradient_Boosting_with_Early_stopping():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)
    errors = [mean_squared_error(y_val, y_pred)
              for y_pred in gbrt.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors)
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)

    min_error = np.min(errors)
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.plot(errors, "b.-")
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
    plt.plot([0, 120], [min_error, min_error], "k--")
    plt.plot(bst_n_estimators, min_error, "ko")
    plt.text(bst_n_estimators, min_error * 1.2, "Minimum", ha="center", fontsize=14)
    plt.axis([0, 120, 0, 0.01])
    plt.xlabel("Number of trees")
    plt.title("Validation error", fontsize=14)

    plt.subplot(122)
    plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)

    #save_fig("early_stopping_gbrt_plot")
    plt.savefig("early_stopping_gbrt_plot")
    plt.show()

    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # early stopping
    print(gbrt.n_estimators)

if __name__ == "__main__":
    '''
    随着你不断投掷硬币， 正面朝上的比率越来越接近于正面的概率（51%） 。 
    图7-3显示了10条偏倚硬币的投掷结果。 可以看出随着投掷次数的增加， 正面的比率逐渐接近51%， 
    最终所有10条线全都接近51%， 并且始终位于50%以上。'''
    #Voting_classifiers()

    # three_classifier()
    '''
    集成预测的泛化效果很可能会比单独的决策树要好一些： 
    二者偏差相近， 但是集成的方差更小（两边训练集上的错误数量差不多， 但是集成的决策边界更规则） '''
    #bagging_ensembles()

    # random_forests()

    # 特征重要性
    #feature_importance()
    #feature_importance2()

    #AdaBoost()
    # AdaBoost2()

    #Gradient_Boosting()
    #lessOrMore()
    Gradient_Boosting_with_Early_stopping()