# encoding: utf-8

import tensorflow as tf
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics
from tensorflow.keras.layers import Dense

N_SAMPLES = 1000  # 采样点数
N_Epochs = 300
TEST_SIZE = 0.3  # 测试数量比率
OUTPUT_DIR = r'F:/DeepLearning_All/Deep-Learning-with-TensorFlow-book-master/ch09'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# 产生一个简单的样本数据集，半环形图，类似的有make_circles,环形数据
X, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=100)  # (1000, 2),(1000, 1)
# 将矩阵随机划分训练集和测试集 (700,2),(300,2),(700,1),(300,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)


def make_plot(X, y, plot_name, file_name, XX=None, YY=None, preds=None):
    plt.figure()
    axes = plt.gca()
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])
    axes.set(xlabel="$x_l$", ylabel="$x_2$")

    # 根据网络输出绘制预测曲面
    # markers = ['o' if i == 1 else 's' for i in y.ravel()]
    # plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, cmap=plt.cm.Spectral, edgecolors='none', m=markers)
    if XX is None and YY is None and preds is None:
        yr = y.ravel()
        for step in range(X[:, 0].size):
            if yr[step] == 1:
                plt.scatter(X[step, 0], X[step, 1], c='b', s=20, cmap=plt.cm.Spectral, edgecolors='none', marker='o')
            else:
                plt.scatter(X[step, 0], X[step, 1], c='r', s=20, cmap=plt.cm.Spectral, edgecolors='none', marker='s')
        plt.savefig(OUTPUT_DIR+'/'+file_name)
        # plt.show()
    else:
        plt.contour(XX, YY, preds, cmap=plt.cm.autumn, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.autumn, edgecolors='k')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决plt.title乱码的问题
        plt.rcParams['axes.unicode_minus'] = False
        plt.title(plot_name)
        plt.savefig(OUTPUT_DIR+'/'+file_name)
        # plt.show()


make_plot(X, y, None, "exam7_dataset.svg")

# 创建网络 5种不同的网络
for n in range(5):
    model = Sequential()  # 创建容器
    model.add(Dense(8, input_dim=2, activation='relu'))  # 第一层
    for _ in range(n):
        model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 创建末尾一层
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 模型的装配
    history = model.fit(X_train, y_train, epochs=N_Epochs, verbose=1)
    # 绘制不同层数的网络决策边界曲线
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    # XX(477, 600), YY(477, 600)
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))  # 创建网格
    Z = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])  # (286200, 1) [0 or 1]
    preds = Z.reshape(XX.shape)
    title = "网络层数({})".format(n)
    file = "网络容量%f.png" % (2+n*1)
    make_plot(X_train, y_train, title, file, XX, YY, preds)