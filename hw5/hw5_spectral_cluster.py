import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        content = f.readlines()
        for i in content:
            i = i[:-1].split(" ")
            data.append([float(i[0]), float(i[1])])
    data = np.array(data)
    return data


def generate_graph(data, k, sigma):
    """
    Parameter:
        data: data to be clustered
        k: number of neighbor
        sigma: parameter of weight in samilarity matrix
    Return:
        W: degree Matrix
    """
    m, n = data.shape
    dist = np.zeros((m, m), dtype=np.float64)
    W = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            dist[i, j] = np.linalg.norm(data[i] - data[j]) ** 2
    # xi's k nearest neighbours
    for i in range(m):
        dist_with_index = zip(dist[i], range(m))
        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
        neighbours_id = [dist_with_index[j][1] for j in range(k+1)]
        for j in neighbours_id:
            if i != j:
                W[i, j] = np.exp(-dist[i, j] / (2 * sigma ** 2))
    W = (W.T + W)/2
    # print(W)
    return W


def Ng_algorithm(W, c):
    """
    Parameter:
        W: degree Matrix
        c: number of classes
    Return:
        label: every data's label like [0,1,1,..,0,1] (n*1)
    """
    # 1.Degree Matrix: D=diag(sum(W))
    W_row_sum = np.sum(W, axis=1)
    D = np.diag(W_row_sum)
    # 2.Laplacian Matrix: L=D-W
    L = D - W
    # 3.normailzed matrix L_sym=D^(-1/2) L D^(-1/2)
    sqrt_D = np.diag(W_row_sum ** (-0.5))
    L_sym = sqrt_D.dot(L).dot(sqrt_D)
    # 4.eigen decomposition
    e_value, e_vector = np.linalg.eig(L_sym)
    e_vector = e_vector.T
    e = zip(e_value, e_vector)
    e_sorted = sorted(e, key=lambda e: e[0])
    # 5.get top-min-eigen-value c vectors
    U = []
    for i in range(c):
        U.append(e_sorted[i][1])
    U = np.array(U).T
    # 6.noemalize new feature
    T = []
    for val in U:
        val = val / np.linalg.norm(U, axis=0)
        T.append(val)
    T = np.array(T)
    # 7.kmeans
    kmeans = KMeans(n_clusters=2)
    label = kmeans.fit_predict(T)
    return label


def main_process(k, sigma):
    W = generate_graph(data, k, sigma)
    c = 2
    label = Ng_algorithm(W, c)
    label_real = []
    for i in range(100):
        label_real.append(0)
    for i in range(100):
        label_real.append(1)
    label_real = np.array(label_real)
    acc = np.sum(label == label_real) / 200
    if acc < 0.5:
        for (idx, val) in enumerate(label):
            if val == 0:
                label[idx] = 1
            else:
                label[idx] = 0
    acc = np.sum(label == label_real) / 200
    print('k: {}\nsigma: {}\nlabel: {}\nacc: {}'.format(k, sigma, label, acc))
    # plt.scatter(data[:, 0], data[:, 1], c=label)
    # plt.show()
    return label, acc


if __name__ == "__main__":
    data = load_data("./data.txt")
    # # data1 for test
    # data1 = []
    # for (idx, val) in enumerate(data):
    #     if(idx < 10)or(idx > 188):
    #         data1.append(val)
    # data1 = np.array(data1)
    # print(data1)

    # Part 1.Plot a predict figure
    label, acc = main_process(5, 0.5)
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.show()

    # Part 2.Plot Acc with different k & sigma
    ks = [5, 15, 25, 35, 45]
    sigmas = np.arange(0.1, 2.5, 0.1)
    for k in ks:
        Acc = []
        for sigma in sigmas:
            label, acc = main_process(k, sigma)
            Acc.append(acc)
        plt.plot(sigmas, Acc)
        plt.title("Acc-Sigma (k={})".format(k))
        plt.xlabel("sigma")
        plt.ylabel("acc")
        plt.show()
