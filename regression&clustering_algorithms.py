from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def mylinearregression():
    """
    线性回归预测房子价格
    :return:
    """
    cf = fetch_california_housing()

    # 对数据集进行划分
    x_train, x_test, y_train, y_test = train_test_split(cf.data, cf.target, test_size=0.3, random_state=24)

    # 需要做标准化处理对于特征值处理
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 使用线性模型进行预测
    # 使用正规方程求解
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    # print("正规方程——权重系数为：\n", lr.coef_)
    # print("正规方程——偏置为：\n", lr.intercept_)

    y_predict = lr.predict(x_test)
    print("LR的均方误差为：\n", mean_squared_error(y_test, y_predict))

    # 梯度下降进行预测
    sgd = SGDRegressor(eta0=0.006)
    sgd.fit(x_train, y_train)

    # print("梯度下降——权重系数为：\n", sgd.coef_)
    # print("梯度下降——偏置为：\n", sgd.intercept_)

    # 怎么评判这两个方法好坏
    y_predict = sgd.predict(x_test)
    print("SGD的均方误差为：\n", mean_squared_error(y_test, y_predict))

    rd = Ridge(max_iter=100000, alpha=3.0)

    rd.fit(x_train, y_train)
    # print("岭回归的权重参数为：", rd.coef_)

    y_predict = rd.predict(x_test)

    print("岭回归的均方误差为：\n", mean_squared_error(y_test, y_predict))

    return None


def logisticregression():
    """
    逻辑回归进行癌症预测
    :return: None
    """
    # 1、读取数据，处理缺失值以及标准化
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']

    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast"
                       "-cancer-wisconsin.data",
                       names=column_name)

    # 删除缺失值
    data = data.replace(to_replace='?', value=np.nan)

    data = data.dropna()

    # 取出特征值
    x = data[column_name[1:10]]

    y = data[column_name[10]]

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 进行标准化

    # 使用逻辑回归
    lr = LogisticRegression()

    lr.fit(x_train, y_train)

    print("得出来的权重：", lr.coef_)

    # 预测类别
    print("预测的类别：", lr.predict(x_test))

    # 得出准确率
    print("预测的准确率:", lr.score(x_test, y_test))
    return None


if __name__ == "__main__":
    # 代码1：线性回归预测房子价格
    # mylinearregression()

    # 代码2：逻辑回归进行癌症预测
    logisticregression()
