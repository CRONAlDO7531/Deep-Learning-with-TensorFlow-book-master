# 创建网络 5种不同数量的Dropout层的网络
for n in range(5):
    model = Sequential()  # 创建容器
    model.add(Dense(8, input_dim=2, activation='relu'))  # 第一层
    counter = 0
    for _ in range(5):  # 网络层数固定为5
        model.add(Dense(64, activation='relu'))
        if counter < n:  # 添加n个Dropout层
            counter += 1
            model.add(layers.Dropout(rate=0.5))
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
    title = "Dropout({})".format(n)
    file = "Dropout%f.png" % (n)
    make_plot(X_train, y_train, title, file, XX, YY, preds)