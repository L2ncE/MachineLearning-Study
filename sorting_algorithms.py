from sklearn.datasets import load_iris
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1 获取数据
    iris = load_iris()

    # 2 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3 特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4 KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    # print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法2：计算准确率
    # 先用训练集造出模型再用测试集来预测并得到准确率
    score = estimator.score(x_test, y_test)
    print("KNN准确率为：\n", score)

    return None


def knn_iris_gscv():
    """
       用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
       :return:
       """
    # 1 获取数据
    iris = load_iris()

    # 2 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3 特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4 KNN算法预估器
    estimator = KNeighborsClassifier()

    # 加入网格搜索和交叉验证
    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # 5 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    # print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法2：计算准确率
    # 先用训练集造出模型再用测试集来预测并得到准确率
    score = estimator.score(x_test, y_test)
    print("KNN加上网格搜索与交叉验证准确率为：\n", score)

    # 最佳参数：best_params_
    # print("最佳参数\n", estimator.best_params_)
    # 最佳结果：best_score_
    # print("最佳结果\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    # print("最佳估计器\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    # print("最佳交叉验证结果\n", estimator.cv_results_)

    return None


def nbcls():
    """
    朴素贝叶斯对新闻数据集进行预测
    :return:
    """
    # 获取新闻的数据，20个类别
    news = fetch_20newsgroups(subset='all')

    # 进行数据集分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3)

    # 对于文本数据，进行特征抽取
    tf = TfidfVectorizer()

    x_train = tf.fit_transform(x_train)
    # 这里打印出来的列表是：训练集当中的所有不同词的组成的一个列表
    # print(tf.get_feature_names())
    # print(x_train.toarray())

    # 不能调用fit_transform
    x_test = tf.transform(x_test)

    # estimator估计器流程
    mlb = MultinomialNB(alpha=1.0)

    mlb.fit(x_train, y_train)

    # 进行预测
    y_predict = mlb.predict(x_test)

    print("预测每篇文章的类别：", y_predict[:100])
    print("真实类别为：", y_test[:100])

    print("预测准确率为：", mlb.score(x_test, y_test))

    return None


def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return:
    """
    # 1. 获取数据集
    iris = load_iris()

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3. 决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4. 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    # print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法2：计算准确率
    # 先用训练集造出模型再用测试集来预测并得到准确率
    score = estimator.score(x_test, y_test)
    print("决策树准确率为：\n", score)
    return None


if __name__ == "__main__":
    # 代码1 KNN算法对鸢尾花进行分类
    # knn_iris()

    # 代码2 网格搜索、交叉验证
    # knn_iris_gscv()

    # 代码3 朴素贝叶斯对新闻数据集进行预测
    # nbcls()

    # 代码4 决策树算法对鸢尾花进行分类
    decision_iris()
