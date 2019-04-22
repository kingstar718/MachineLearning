import numpy as np
import matplotlib.pyplot as plt

#LinearSVR类也是LinearSVC类的回归等价物
def linear_SVM_reg():
    np.random.seed(42)
    m = 50
    X = 2 * np.random.rand(m, 1)
    y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

    from sklearn.svm import LinearSVR
    svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
    svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)
    svm_reg1.fit(X, y)
    svm_reg2.fit(X, y)

    def find_support_vectors(svm_reg, X, y):
        y_pred = svm_reg.predict(X)
        off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
        return np.argwhere(off_margin)

    svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
    svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

    eps_x1 = 1
    eps_y_pred = svm_reg1.predict([[eps_x1]])

    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    # plt.plot([eps_x1, eps_x1], [eps_y_pred, eps_y_pred - svm_reg1.epsilon], "k-", linewidth=2)
    plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )
    plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
    plt.subplot(122)
    plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
    plt.savefig("svm_regression_plot")
    plt.show()


def plot_svm_regression(svm_reg, X, y, axes):
        x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
        y_pred = svm_reg.predict(x1s)
        plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
        plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
        plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
        plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
        plt.plot(X, y, "bo")
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.legend(loc="upper left", fontsize=18)
        plt.axis(axes)

# SVR类是SVC类的回归等价物
def svr_SVM_reg():
    np.random.seed(42)
    m = 100
    X = 2 * np.random.rand(m, 1) - 1
    y = (0.2 + 0.1 * X + 0.5 * X ** 2 + np.random.randn(m, 1) / 10).ravel()

    from sklearn.svm import SVR

    svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
    svm_poly_reg1.fit(X, y)
    svm_poly_reg2.fit(X, y)

    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon),
              fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.subplot(122)
    plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon),
              fontsize=18)
    plt.savefig("svm_with_polynomial_kernel_plot")
    plt.show()

#鸢尾花数据集的决策函数
def iris_SVM():
    from sklearn import datasets
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

    scaler = StandardScaler()
    svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
    svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

    scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
    scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

    scaled_svm_clf1.fit(X, y)
    scaled_svm_clf2.fit(X, y)

    from mpl_toolkits.mplot3d import Axes3D

    def plot_3D_decision_function(ax, w, b, x1_lim=[4, 6], x2_lim=[0.8, 2.8]):
        x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])
        X_crop = X[x1_in_bounds]
        y_crop = y[x1_in_bounds]
        x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
        x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
        x1, x2 = np.meshgrid(x1s, x2s)
        xs = np.c_[x1.ravel(), x2.ravel()]
        df = (xs.dot(w) + b).reshape(x1.shape)
        m = 1 / np.linalg.norm(w)
        boundary_x2s = -x1s * (w[0] / w[1]) - b / w[1]
        margin_x2s_1 = -x1s * (w[0] / w[1]) - (b - 1) / w[1]
        margin_x2s_2 = -x1s * (w[0] / w[1]) - (b + 1) / w[1]
        ax.plot_surface(x1s, x2, 0, color="b", alpha=0.2, cstride=100, rstride=100)
        ax.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
        ax.plot(x1s, margin_x2s_1, 0, "k--", linewidth=2, label=r"$h=\pm 1$")
        ax.plot(x1s, margin_x2s_2, 0, "k--", linewidth=2)
        ax.plot(X_crop[:, 0][y_crop == 1], X_crop[:, 1][y_crop == 1], 0, "g^")
        ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
        ax.plot(X_crop[:, 0][y_crop == 0], X_crop[:, 1][y_crop == 0], 0, "bs")
        ax.axis(x1_lim + x2_lim)
        ax.text(4.5, 2.5, 3.8, "Decision function $h$", fontsize=15)
        ax.set_xlabel(r"Petal length", fontsize=15)
        ax.set_ylabel(r"Petal width", fontsize=15)
        ax.set_zlabel(r"$h = \mathbf{w}^t \cdot \mathbf{x} + b$", fontsize=18)
        ax.legend(loc="upper left", fontsize=16)

    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    plot_3D_decision_function(ax1, w=svm_clf2.coef_[0], b=svm_clf2.intercept_[0])

    plt.savefig("iris_3D_plot")
    plt.show()

def California_housing_SVM():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(data_home="D:\Git\MachineLearning\ML\mldata\cal_housing.tgz")  #
    X = housing["data"]
    y = housing["target"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.svm import LinearSVR

    lin_svr = LinearSVR(random_state=42)
    lin_svr.fit(X_train_scaled, y_train)

    from sklearn.metrics import mean_squared_error

    y_pred = lin_svr.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    print(mse)

if __name__=="__main__":
    #linear_SVM_reg()

    #svr_SVM_reg()

    # iris_SVM()
    California_housing_SVM()
    pass