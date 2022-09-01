from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mylinearregression():
    """
    线性回归预测房子价格
    :return:
    """
    lb = load_boston()

    # 对数据集进行划分
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.3, random_state=24)

    # 需要做标准化处理对于特征值处理
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 对于目标值进行标准化
    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)
    y_test = std_y.inverse_transform(y_test)

    # 使用线性模型进行预测
    # 使用正规方程求解
    lr = LinearRegression()

    lr.fit(x_train, y_train)
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))

    print(lr.coef_)
    print("正规方程预测的结果为：", y_lr_predict)
    print("正规方程的均方误差为：", mean_squared_error(y_test, y_lr_predict))

    # 梯度下降进行预测
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)

    print("SGD的权重参数为：", sgd.coef_)
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("SGD的预测的结果为：", y_sgd_predict)
    # 怎么评判这两个方法好坏
    print("SGD的均方误差为：", mean_squared_error(y_test, y_sgd_predict))

    return None


if __name__ == "__main__":
    # 代码1： 线性回归预测房子价格
    mylinearregression()
