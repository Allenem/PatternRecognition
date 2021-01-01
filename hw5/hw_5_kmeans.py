import numpy as np
import matplotlib.pyplot as plt

# generate 5 kinds of data, plot & return them


def generate_data():
    Sigma = [[1, 0], [0, 1]]
    mu1 = [1, -1]
    mu2 = [5.5, -4.5]
    mu3 = [1, 4]
    mu4 = [6, 4.5]
    mu5 = [9, 0.0]

    x1 = np.random.multivariate_normal(mu1, Sigma, 200)
    x2 = np.random.multivariate_normal(mu2, Sigma, 200)
    x3 = np.random.multivariate_normal(mu3, Sigma, 200)
    x4 = np.random.multivariate_normal(mu4, Sigma, 200)
    x5 = np.random.multivariate_normal(mu5, Sigma, 200)

    # (5*200*2)
    X = np.array([x1, x2, x3, x4, x5])
    colors = ['red', 'blue', 'black', 'green', 'purple']
    for (idx, val) in enumerate(X):
        plt.scatter(val[:, 0], val[:, 1], marker='.', color=colors[idx])
    plt.title('Data')
    plt.show()
    # obtain the 1000 data points to be clustered (1000*2)
    X = np.concatenate((x1, x2, x3, x4, x5), axis=0)
    mu_real = np.array([np.mean(x1, axis=0), np.mean(x2, axis=0), np.mean(
        x3, axis=0), np.mean(x4, axis=0), np.mean(x5, axis=0)])
    label_real = np.zeros(1000)
    lb_idx = 0
    for i in np.arange(5):
        for j in np.arange(200):
            label_real[lb_idx] = i
            lb_idx += 1
    return X, mu_real, label_real


def k_means(data, mu_init):
    """
    Parameters:
        data: data to be clustered (n*d)
        mu_init: initialized means (c*d)
    Return:
        res: cluster result like [[datas in class 0], ... , [datas in class c-1]] (c*about200*d)
        label: every data's label like [0,2,1,..,0,1] (n*1)
        mu: finally mean like [[center of class 0], ... , [center in class c-1]] (c*d)
        cnt: the times of iretation
    """
    mu_old = np.zeros_like(mu_init)
    mu = mu_init
    cnt = 0
    c = len(mu_init)
    n, d = data.shape
    distance = np.zeros((n, c), dtype=np.float64)
    label = np.zeros(len(data))

    while np.sum(mu - mu_old):
        mu_old = mu
        cnt += 1
        # compute distance matrix (n*c)
        for i in range(n):
            for j in range(c):
                distance[i][j] = np.linalg.norm(data[i] - mu[j])
        # compute res(c*about200*d) & label(n*1)
        res = []
        for _ in range(c):
            res.append([])
        for idx, sample in enumerate(data):
            label[idx] = np.argmin(distance[idx])
            res[np.argmin(distance[idx])].append(sample)
        res = np.array(res)
        # recompute class center mu
        mu = []
        for i in res:
            mu.append(np.mean(i, axis=0))
        mu = np.array(mu)
        # print(mu, mu.shape)

    return res, label, mu, cnt


if __name__ == '__main__':
    # 1.generate data
    X, mu_real, label_real = generate_data()
    # 2.generate mu_random & reorder
    # -----------------
    # # mu_random METHOD1
    # X_max = max(X.tolist())
    # X_min = min(X.tolist())
    # mu_random = np.random.uniform(X_max, X_min, (5, 2))
    # -----------------
    # # mu_random METHOD2
    mu_1 = np.array([0.5, -1.0])
    mu_2 = np.array([3.8, -6.5])
    mu_3 = np.array([-1.1, 6.4])
    mu_4 = np.array([5.7, 5.5])
    mu_5 = np.array([7.8, 1.5])

    # mu_1 = np.array([3.3, -2.1])
    # mu_2 = np.array([7.6, -3.2])
    # mu_3 = np.array([13.2, -1.3])
    # mu_4 = np.array([5.4, 6.3])
    # mu_5 = np.array([0.5, 7.2])

    # mu_1 = np.array([0.5, -4.3])
    # mu_2 = np.array([3.8, -6.5])
    # mu_3 = np.array([-3.1, 6.4])
    # mu_4 = np.array([0.7, 5.5])
    # mu_5 = np.array([1.5, 7.8])

    mu_random = np.array([mu_1, mu_2, mu_3, mu_4, mu_5])
    # -----------------
    mu_reorder = np.zeros_like(mu_random)
    # -----------------
    # # reorder METHOD1
    # for (index, value) in enumerate(mu_real):
    #     dist = np.zeros(len(mu_random))
    #     for (idx, val) in enumerate(mu_random):
    #         dist[idx] = np.linalg.norm(mu_real[index] - mu_random[idx])
    #     mu_reorder[index] = mu_random[np.argmin(dist)]
    # -----------------
    # # reorder METHOD2
    idxs = [0, 1, 2, 3, 4]
    for (i, value) in enumerate(mu_random):
        dist = np.zeros(len(mu_real))
        for (j, val) in enumerate(mu_real):
            dist[j] = np.linalg.norm(mu_random[i] - mu_real[j])
        idxs[i] = np.argmin(dist)
    print(idxs)
    for (i, val) in enumerate(idxs):
        mu_reorder[val] = mu_random[i]
    # -----------------
    print('mu_real：{}\nmu_reorder：{}'.format(mu_real, mu_reorder))
    # 3.kmeans
    result, label, mu, cnt = k_means(X, mu_reorder)
    print('label: omit\nmu: {}\niretations: {}'.format(mu, cnt))
    # 4.plot figure & compute accuracy, error
    colors = ['red', 'blue', 'black', 'green', 'purple']
    for (idx, val) in enumerate(result):
        val = np.array(val)
        plt.scatter(val[:, 0], val[:, 1], marker='.',
                    c=colors[idx % len(colors)])
        plt.scatter(mu[:, 0], mu[:, 1], marker='X')
    plt.title('K_means Result')
    plt.show()
    accuracy = np.sum(label == label_real) / 1000
    print('accuracy is %.5f' % accuracy)
    error = (mu - mu_real)
    print('error is {}'.format(error))
