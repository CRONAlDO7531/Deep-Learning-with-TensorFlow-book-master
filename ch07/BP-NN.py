  # encoding: utf-8

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
# from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

N_SAMPLES = 2000  # 采样点数
TEST_SIZE = 0.3  # 测试数量比率

# 产生一个简单的样本数据集，半环形图，类似的有make_circles,环形数据
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)  # (2000, 2),(2000, 1)
# X, y = make_circles(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# 将矩阵随机划分训练集和测试集 (1400,2),(600,2),(1400,1),(600,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)

# 绘制数据集分布，X为2D坐标，y为数据点标签

def make_plot(X, y, plot_name=None, XX=None, YY=None, preds=None, dark=False):
    if dark:
        plt.style.use('dark_background')
    else:
        sns.set_style('whitegrid')
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$x_l$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)  # 调整边距和子图间距，子图的左侧
    plt.subplots_adjust(right=0.80)
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.shape(XX.shape), 25, alpha=1, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[1.5], cmap="Greys", vmin=0, vmax=.6)
    # 根据标签区分颜色
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')

    plt.savefig('data_set.png')
    plt.close()


make_plot(X, y, "Classification DataSet Visualization")
plt.show()


class Layer:
    # 全连接层网络
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        : int n_input: 输入节点数
        ：int n_neurons: 输出节点数
        ：str activation: 激活函数类型
        : weights: 权值张量，内部生成
        ： bias: 偏置，内部生成
        """
        # 通过正态分布生成初始化的参数
        self.weights \
            = weights if weights is not None else \
            np.random.randn(n_input, n_neurons) * np.sqrt(1/n_neurons)
        self.bias \
            = bias if bias is not None else \
            np.random.randn(n_neurons) * 0.1
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None

    # 网络的前向传播
    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias  # X@w + b
        self.last_activation = self._apply_activation(r)  # 激活函数
        return self.last_activation

    # 不同类型的激活函数
    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

        # 不同类型激活函数的导数实现
    def apply_activation_derivation(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
             return 1 - r**2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r


# 神经网络模型
class NeuralNetwork:
    def __init__(self):  # 需要实例化后对属性赋值
        self._layers = []  # 网络层对象列表

    def add_layer(self, layer):  # 追加网络层
        self._layers.append(layer)

    # 前向传播只需要循环调用各网络层对象的前向计算函数
    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    # 网络模型的反向传播
    def backpropagation(self, X, y, learning_rate):
        output = self.feed_forward(X)
        # 反向循环
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]  # 得到当前层对象
            if layer == self._layers[-1]:  #如果是输出层
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivation(output)
            else:  # 计算隐藏层
                next_layer = self._layers[i + 1]  # 得到下一层对象
                layer.error = np.dot(next_layer.weights, next_layer.delta)  # 矩阵乘法
                layer.delta = layer.error *\
                              layer.apply_activation_derivation(layer.last_activation)

        for i in range(len(self._layers)):
            layer = self._layers[i]
            # o_i为上一层网络输出
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)  # 将数据视为2维数据
            layer.weights += layer.delta * o_i.T * learning_rate  # .T是转置

    # 网络的训练
    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        temp1 = y_train.shape[0]
        y_onehot = np.zeros((temp1, 2))
        temp2 = np.arange(y_train.shape[0])  # 线性 0 - 1399
        y_onehot[temp2, y_train] = 1
        mses = []
        accuracy = []
        for i in range(max_epochs):
            for j in range(len(X_train)):  # 一次训练一个样本
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
                acc = self.accuracy(self.predict(X_test), y_test.flatten())
                print('Accuracy: %.2f%%' % (acc * 100))
                accuracy.append(acc*100)
        return mses, accuracy

    def accuracy(self, y_output, y_test):
        return np.mean((np.argmax(y_output, axis=1) == y_test))

    def predict(self, X_test):
        return self.feed_forward(X_test)


# 4层全连接网络 实例化训练和预测
nn = NeuralNetwork()  # 实列化网络
nn.add_layer(Layer(2, 25, 'sigmoid'))  # 2 --> 25
nn.add_layer(Layer(25, 50, 'sigmoid'))  # 25 --> 50
nn.add_layer(Layer(50, 25, 'sigmoid'))  # 50 --> 25
nn.add_layer(Layer(25, 2, 'sigmoid'))  # 25 --> 2
learning_rate = 0.01
max_epochs = 1000
mses, accuracy = nn.train(X_train, X_test, y_train, y_test, learning_rate, max_epochs)

plt.figure()
plt.plot(mses, 'b', label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig('exam5.2 MSE Loss.png')
plt.show()

plt.figure()
plt.plot(accuracy, 'r', label='Accuracy rate')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('exam5.2 Accuracy.png')
plt.show()