import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('data.txt')
X = data[:, 0:2]
ten_ones = np.ones((10, 1), dtype=float)
W1_X = X[0:10]  # 10*2
W2_X = X[10:20]
W3_X = X[20:30]
W4_X = X[30:40]
W1_Y = np.c_[ten_ones, W1_X]  # 10*3
W2_Y = np.c_[ten_ones, W2_X]
W3_Y = np.c_[ten_ones, W3_X]
W4_Y = np.c_[ten_ones, W4_X]


# Batch perceptron algorithm
def Batch_perceptron(Wm_Y, Wn_Y):
    Y = np.r_[Wm_Y, -Wn_Y]  # 20*3
    Errs = list(Y)
    a = np.zeros((3, 1), dtype=float)
    k = 0
    learnrate = 0.1
    while len(Errs) != 0:
        Errs = []
        for i in range(len(Y)):
            if np.dot(Y[i], a)[0] <= 0:
                Errs.append(Y[i])
        a += learnrate * np.array([np.array(Errs).sum(axis=0)]).T
        k += 1
        if k > 10000:
            print('Batch_perceptron 算法不能拟合{}&{}的分界面'.format(Wm_Y, Wn_Y))
            return a.flatten()
    print('Batch_perceptron algorithm：\na = {}\nIterations required for convergence is {}'.format(
        a.flatten(), k))
    return a.flatten()


# Ho-Kashyap algorithm
def Ho_Kashyap(Wm_Y, Wn_Y):
    Y = np.r_[Wm_Y, -Wn_Y]
    n, d = Y.shape
    # a = np.zeros((d, 1))
    a = np.linalg.inv(Y.T.dot(Y)).dot(Y.T).dot(np.ones((n, 1)))
    b = np.ones((n, 1)) * 1e-2
    learnrate = 0.1
    k = 0
    bmin = 1e-10
    kmax = 100000
    while k < kmax:
        e = Y.dot(a) - b
        eplus = .5 * (e + np.abs(e))
        b += 2 * learnrate * eplus
        Yplus = np.linalg.inv(Y.T.dot(Y)).dot(Y.T)
        a = Yplus.dot(b)
        k += 1
        if np.abs(e).all() <= bmin:
            print('Ho-Kashyap algorithm：\na = {}\nb = {}\nIterations required for convergence is {}'.format(
                a, b, k))
            return a.flatten(), b.flatten()
    print("Ho-Kashyap algorithm：\nNo solution found!")
    return a.flatten(), b.flatten()


# MSE准则多分类
def MSE():
    Y_train = np.c_[W1_Y[0:8].T, W2_Y[0:8].T, W3_Y[0:8].T, W4_Y[0:8].T]  # 3*32
    Y_test = np.c_[W1_Y[-2:].T, W2_Y[-2:].T, W3_Y[-2:].T, W4_Y[-2:].T]  # 3*8
    # label_train = ...
    # label_test = [[11000000],[00110000],[00001100],[00000011]]
    label_train = np.zeros([4, 32])
    label_test = np.zeros([4, 8])
    for i in range(len(label_train)):
        for j in range(len(label_train[0])):
            label_train[i, j] = int(int(j / 8) == i)
    for i in range(len(label_test)):
        for j in range(len(label_test[0])):
            label_test[i, j] = int(int(j / 2) == i)
    a = np.linalg.pinv(Y_train.T).dot(
        label_train.T).T  # 广义逆矩阵 3*32, 3*32*32*4.T = 4*3
    target = np.arange(1, 5).dot(label_test)  # [11223344]
    output = np.argmax(a.dot(Y_test), axis=0) + np.ones(8)  # 4*8->1*8
    accuracy = sum(output == target) / len(target)
    print('MSE准则多分类：\ntarget = {}\noutput = {}\naccuracy = {}'.format(
        target, output, accuracy))


# Plot figure
def Plot_figure(nums, a):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('X')
    plt.ylabel('Y')
    x1 = W1_X[:, 0]
    y1 = W1_X[:, 1]
    x2 = W2_X[:, 0]
    y2 = W2_X[:, 1]
    x3 = W3_X[:, 0]
    y3 = W3_X[:, 1]
    x4 = W4_X[:, 0]
    y4 = W4_X[:, 1]
    x = [x1, x2, x3, x4]
    y = [y1, y2, y3, y4]
    area = np.pi * 2 ** 2
    colors = ['red', 'green', 'purple', 'blue']
    if a[2] != 0:
        xd = np.arange(-10, 10, 0.01)
        yd = -a[0]/a[2]-a[1]*xd/a[2]
        plt.plot(xd, yd, linewidth='0.5', color='#000000')
    for num in nums:
        plt.scatter(x[int(num)-1], y[int(num)-1], s=area, c=colors[int(num)-1],
                    alpha=0.4, label='w'+num)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    a = Batch_perceptron(W1_Y, W2_Y)
    Plot_figure(['1', '2'], a)
    a = Batch_perceptron(W3_Y, W2_Y)
    Plot_figure(['3', '2'], a)
    a, b = Ho_Kashyap(W1_Y, W3_Y)
    Plot_figure(['1', '3'], a)
    a, b = Ho_Kashyap(W2_Y, W4_Y)
    Plot_figure(['2', '4'], a)
    MSE()

# OUTPUT
#
# Batch_perceptron algorithm：
# a = [ 3.4  -3.04  3.41]
# Iterations required for convergence is 24
# Batch_perceptron algorithm：
# a = [ 1.9  -4.14  4.86]
# Iterations required for convergence is 17
# Ho-Kashyap algorithm：
# No solution found!
# Ho-Kashyap algorithm：
# a = [[0.13381828]
#  [0.03717747]
#  [0.01669066]]
# b = [[0.46787909]
#  [0.01      ]
#  [0.30111691]
#  [0.3947414 ]
#  [0.32167591]
#  [0.13245664]
#  [0.15628159]
#  [0.12494896]
#  [0.50786447]
#  [0.24952647]
#  [0.08073817]
#  [0.19372309]
#  [0.15084515]
#  [0.23560033]
#  [0.18203341]
#  [0.03832449]
#  [0.17504972]
#  [0.29644592]
#  [0.29204113]
#  [0.26875263]]
# Iterations required for convergence is 29132
# MSE准则多分类：
# target = [1. 1. 2. 2. 3. 3. 4. 4.]
# output = [1. 1. 2. 2. 3. 3. 4. 4.]
# accuracy = 1.0
