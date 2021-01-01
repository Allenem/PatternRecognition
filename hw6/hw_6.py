import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

# train & train_label file path
train_images_idx3_ubyte_file = './MNIST_data/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './MNIST_data/train-labels.idx1-ubyte'
# test & test_label file path
test_images_idx3_ubyte_file = './MNIST_data/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './MNIST_data/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    Decode images files(idx3 type)
    Parameter:
        idx3_ubyte_file: idx3 file path
    Return:
        images: image dataset
    """
    # 1.Parse header information: magic_number, image_number, image_height, image_weight
    # We need 4i which means first 4 rows 32 int in images, and we need 2i in labels.
    fmt_header = '>4i'
    # Read idx3 files
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)
    print('magic_number: %d, image_number: %d, image_size: %d*%d' %
          (magic_number, num_images, num_rows, num_cols))
    # 2.Parse data set
    image_size = num_rows * num_cols
    # The type of image pixel is unsigned char, format is B, we need read 784 B datas.
    fmt_image = '>' + str(image_size) + 'B'
    # Pointer's offset = 0016
    offset += struct.calcsize(fmt_header)
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(
            fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
        if (i + 1) == num_images:
            print('Number of parsed images: %d' % (i+1))
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    Decode labels files(idx1 type)
    Parameter:
        idx1_ubyte_file: idx1 file path
    Return:
        labels: label dataset
    """
    # 1.Parse header information: magic_number, image_number
    # We need 2i which means first 2 rows 32 int in images
    fmt_header = '>2i'
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number: %d, image_number: %d' % (magic_number, num_images))
    # 2.Parse data set
    fmt_image = '>B'
    offset += struct.calcsize(fmt_header)
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
        if (i + 1) == num_images:
            print('Number of parsed labels: %d' % (i+1))
    return labels


def data_load(cls=[0, 1], scale=1):
    # 1.Parse data, size: n*row*col, n*1, n*row*col, n*1
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
    print(train_images.shape)
    # 2.Use label is 0 & 1 data
    # (12665, 28, 28)
    train_images_temp1 = np.concatenate(
        (train_images[train_labels == cls[0]], train_images[train_labels == cls[1]]), axis=0)
    train_labels_temp1 = np.concatenate(
        [train_labels[train_labels == cls[0]], train_labels[train_labels == cls[1]]], axis=0)
    test_images_temp1 = np.concatenate(
        [test_images[test_labels == cls[0]], test_images[test_labels == cls[1]]], axis=0)
    test_labels_temp1 = np.concatenate(
        [test_labels[test_labels == cls[0]], test_labels[test_labels == cls[1]]], axis=0)
    # (12665, 784)
    train_images_temp2 = np.reshape(
        train_images_temp1, (train_images_temp1.shape[0], -1))
    test_images_temp2 = np.reshape(
        test_images_temp1, (test_images_temp1.shape[0], -1))
    # 3.Transform train_images & test_images features by scaling each feature to a given range [0,1]
    if scale:
        # MinMaxScaler：Transform features by scaling each feature to a given range [0,1]
        # MaxAbsScaler：Transform features by scaling each feature to a given range [-1,1]
        scaler = MinMaxScaler()
        train_images_temp3 = scaler.fit_transform(train_images_temp2)
        test_images_temp3 = scaler.transform(test_images_temp2)

    return train_images_temp3, train_labels_temp1, test_images_temp3, test_labels_temp1, train_images_temp1


if __name__ == '__main__':
    # 1.Get data
    classes = [0, 2]
    train_images, train_labels, test_images, test_labels, train_images_temp1 = data_load(
        cls=classes)
    print('Train images number: {}\nTest images number: {}'.format(
        len(train_labels), len(test_labels)))

    # # 2.Verify label & image
    # for i in range(10):
    #     print(train_labels[5918+i])
    #     plt.imshow(train_images_temp1[5918+i], cmap='gray')
    #     plt.show()
    # print('done')

    # 3.SVM with different c & gamma
    Cs = [0.001, 0.1, 1, 1e6]
    gammas = [1/80, 1/800, 1/800000, 0]
    for C in Cs:
        Acc1 = []
        Acc2 = []
        for gamma in gammas:
            print('C = %f, gamma = %f' % (C, gamma))
            # 3.1.New a SVM predictor
            if gamma == 0:
                predictor = svm.SVC(gamma='auto', C=C,
                                    kernel='linear', max_iter=100)
            else:
                predictor = svm.SVC(
                    gamma=gamma, C=C, kernel='rbf', max_iter=100)
            # 3.2.Train SVM model
            predictor.fit(train_images, np.int8(train_labels))
            # 3.3.Get accuracy of trainset & testset
            result1 = predictor.score(train_images, train_labels)
            print('The accuracy of trainset is %f' % result1)
            result2 = predictor.score(test_images, test_labels)
            print('the accuracy of testset is %f' % result2)
            Acc1.append(round(result1, 4))
            Acc2.append(round(result2, 4))
        # 3.4.plot Acc-Gammas with different C
        # plt.semilogx(gammas, Acc1)
        plt.plot(gammas, Acc1)
        for a, b in zip(gammas, Acc1):
            plt.text(a, b, (a, b), ha='center', va='bottom')
        # plt.semilogx(gammas, Acc2)
        plt.plot(gammas, Acc2)
        for a, b in zip(gammas, Acc2):
            plt.text(a, b, (a, b), ha='center', va='bottom')
        plt.legend(['Train Acc', 'Test Acc'])
        plt.title("Acc-Gammas (C={})".format(C))
        plt.xlabel("gamma")
        plt.ylabel("acc")
        plt.show()
