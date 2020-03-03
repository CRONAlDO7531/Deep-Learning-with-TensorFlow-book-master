#%%
import  matplotlib
from    matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False 
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers
import  os





os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)


def preprocess(x, y): 
    # [b, 28, 28], [b]
    print(x.shape,y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x,y

#%%
(x, y), (x_test, y_test) = datasets.mnist.load_data()  #加载数据 分配张量
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
#%%
batchsz = 512  #一次并行计算样本数据的数量
train_db = tf.data.Dataset.from_tensor_slices((x, y))  #将训练集数据转换成 Dataset 对象
train_db = train_db.shuffle(1000)  #随机打散
train_db = train_db.batch(batchsz)  #批量运算
train_db = train_db.map(preprocess)  #调用与处理函数，对数据进行预处理
train_db = train_db.repeat(20)   #将数据重复使用20次

#%%

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  #将测试集数据转换成 Dataset 对象
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)   #随机打散 批量运算 进行预处理
x,y = next(iter(train_db))  #？迭代器  迭代赋值
print('train sample:', x.shape, y.shape)
# print(x[0], y[0])




#%%
def main():

    # learning rate
    lr = 1e-2
    accs,losses = [], []   #精准度和误差


    # 784 => 512      W1的张量存储 ，B1的张量存储  输入层-隐含层1
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 512 => 256      隐含层1-隐含层2
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 256 => 10       隐含层2-输出层
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))



 

    for step, (x,y) in enumerate(train_db): #enumerate用于遍历整个训练集 如果（x,y）在数据集中则进行循环 step为循环次数
 
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))  #改变张量维度 便于数据直接输入网络的输入层-数据对接输入层

        with tf.GradientTape() as tape:  #引用梯度求解利器  tape.graientTape()对目标函数进行求导

            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)  #调用激活函数relu()
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y-out)  #计算误差的平方
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)  #求误差的平均值    即两步求得均方误差

 
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3]) 
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):  #组合成以g命名的新的list
            p.assign_sub(lr * g)  #所有参数乘学习率


        # print
        if step % 80 == 0:
            print(step, 'loss:', float(loss))
            losses.append(float(loss))
 
        if step %80 == 0:
            # evaluate/test
            total, total_correct = 0., 0  #新变量大的定义方法

            for x, y in test_db:
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)  #在向量行中找到最大值的索引
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)   #在向量行中找到最大值的索引  axis=0时 找到向量列中最大值的索引
                # bool type
                correct = tf.equal(pred, y)   #判断预测值和测试值是否相等
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()  #总正确率 统计正确的个数
            total += x.shape[0]      #tf.reduce_sum-求和函数  tf.cast-数据类型转换

            print(step, 'Evaluate Acc:', total_correct/total)  #输出准确率  正确的除以总数

            accs.append(total_correct/total)


    plt.figure()
    x = [i*80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')

if __name__ == '__main__':
    main()