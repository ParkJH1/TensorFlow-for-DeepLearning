import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt

from six.moves import urllib
import os
import gzip

tf = tf_new.compat.v1
tf.set_random_seed(777)


def mnist_load(num_images=100):
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    filepaths = []
    for filename in filenames:
        filepath = os.path.join('data', filename)
        filepaths.append(filepath)
        if not os.path.exists(filepath):
            print('Downloading...', filename)
            filepath, _ = urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + filenames[i], filepath)
            size = os.stat(filepath).st_size
            print('Successfully downloaded', filename, size, 'bytes.')
        else:
            print('Already downloaded', filename)

    print('Extracting...', filenames[0])
    with gzip.open(filepaths[0]) as bytestream:
        size = os.stat(filepaths[0]).st_size
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(-1, 28, 28)
        print('Successfully Extracted', filenames[0])

    print('Extracting...', filenames[1])
    with gzip.open(filepaths[1]) as bytestream:
        size = os.stat(filepaths[1]).st_size
        bytestream.read(8)
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        print('Successfully Extracted', filenames[1])

    return images, labels


x_data, y_data = mnist_load(9)
print(x_data)
print(y_data)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_data[i], cmap='gray')
    plt.title(str(int(y_data[i])))
    plt.xticks([])
    plt.yticks([])

plt.show()
