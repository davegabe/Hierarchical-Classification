import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_pre_path = 'dataset/cifar-100-python/' # change this path
# File paths
data_train_path = data_pre_path + 'train'
data_test_path = data_pre_path + 'test'
# Read dictionary
data_train_dict = unpickle(data_train_path)

data_train = data_train_dict[b'data']

# Reshape the data
data_train = data_train.reshape((len(data_train), 3, 32, 32))

# Save each image
for i in range(len(data_train)):
    img = data_train[i]
    img = np.moveaxis(img, 0, 2)
    img = Image.fromarray(img)
    img.save('dataset/cifar-100-images/' + str(i) + '.png')