import jieba
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    # print("鸢尾花数据集：\n", iris)
    # print("查看数据集描述：\n", iris["DESCR"])
    # print("查看特征值的名字：\n", iris.feature_names)
    # print("查看特征值：\n", iris.data, iris.data.shape)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)

    return None


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=False)
    # 2、调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


def count_demo():
    """
    文本特征抽取：CountVectorizer
    :return:
    """
    data = ["life is short, i like like python", "life is too long, i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer()

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def cut_word(text):
    """
    进行中文分词："我爱北京天安门" ——> "我 爱 北京 天安门"
    :param text:
    :return:
    """
    text = " ".join(jieba.cut(text))
    return text


def count_chinese_demo():
    """
    中文文本特征抽取
    :return:
    """
    # 1、将中文文本进行分词
    data = [
        "就像半藏说的那样，我个人对于这款游戏的期待不在战斗系统中，我们国家不可能一口气做出个世界第一 。",
        "但是剧情以及场景，基于题材可以融合许多古代建筑、文物、志怪文化以及许多古诗词，不提国外玩家，对于国内市场，",
        "玩家中的文化交流、发售后的各类科普、精美周边的发售，何尝不是又一场西游梦呢。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # 2、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一场", "可以"])

    # 3、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


if __name__ == "__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()

    # 代码2：字典特征抽取
    # dict_demo()

    # 代码3：文本特征抽取
    # count_demo()

    # 代码5：中文文本特征抽取
    count_chinese_demo()
