import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt

def KNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must be equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    # 获取x和所有X_train的距离
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    # 将distances排序，获得排好序的元素的索引
    nearest = np.argsort(distances)
    # 获取最近的k个元素的索引，并获取这些元素的标记
    topK_y = [y_train[i] for i in nearest[:k]]
    # 投票
    votes = Counter(topK_y)
    # 返回预测结果
    return votes.most_common(1)[0][0]
