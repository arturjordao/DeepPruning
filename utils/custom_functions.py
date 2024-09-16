import random

import numpy as np


def load_transformer_model(architecture_file='', weights_file=''):
    """
    loads a premade transformer model from a set of files
    Args:
        architecture_file: name of the file containing architecture of the model
        weights_file: name of the file containing weights of the model
    Returns:
        loaded transformer model
    """
    import keras
    from custom_classes import Patches, PatchEncoder
    from keras.utils import CustomObjectScope

    if '.json' not in architecture_file:
        architecture_file = architecture_file + '.json'

    with open(architecture_file, 'r') as f:
        with CustomObjectScope({'PatchEncoder': PatchEncoder},
                               {'Patches': Patches}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model


def load_model(architecture_file='', weights_file=''):
    """
    loads a premade neural network model from a set of files
    Args:
        architecture_file: name of the file containing architecture of the model
        weights_file: name of the file containing weights of the model
    Returns:
        loaded neural network model
    """
    import keras
    from keras.utils.generic_utils import CustomObjectScope
    from keras import backend as K
    from keras import layers

    def _hard_swish(x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _relu6(x):
        return K.relu(x, max_value=6)

    if '.json' not in architecture_file:
        architecture_file = architecture_file + '.json'

    with open(architecture_file, 'r') as f:
        # Not compatible with keras 2.4.x and TF 2.0
        # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
        #                         'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D,
        #                        '_hard_swish': _hard_swish}):
        with CustomObjectScope({'relu6': _relu6,
                                'DepthwiseConv2D': layers.DepthwiseConv2D,
                                '_hard_swish': _hard_swish}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model


def save_model(file_name='', model=None):
    """
    saves a model into architecture and weights file
    Args:
        file_name: name of the file the model is saved into
        model: model to be saved
    """
    print('Saving architecture and weights in {}'.format(file_name))

    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())


def generate_data_augmentation(x_train):
    print('Using real-time data augmentation.')
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)
    return datagen


def cifar_resnet_data(debug=True, validation_set=False):
    import tensorflow as tf
    print('Debuging Mode') if debug is True else print('Real Mode')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    if debug:
        # random_train = random.sample(range(200), 40)
        # idx_train = random_train[0:31]
        # idx_test = random_train[31:41]

        idx_train = [4, 5, 32, 6, 24, 41, 38, 39, 59, 58, 28, 20, 27, 40, 51, 95, 103, 104, 84, 85, 87, 62, 8, 92, 67,
                     71, 76, 93, 129, 76]
        idx_test = [9, 25, 0, 22, 24, 4, 20, 1, 11, 3]

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    # y_test = np.argmax(y_test, axis=1)

    if validation_set is False:
        return x_train, y_train, x_test, y_test
    else:
        datagen = generate_data_augmentation(x_train)
        for x_val, y_val in datagen.flow(x_train, y_train, batch_size=5000):
            break
        return x_train, y_train, x_test, y_test, x_val, y_val


def count_filters(model):
    import keras
    # from keras.applications.mobilenet import DepthwiseConv2D
    from keras.layers import DepthwiseConv2D
    n_filters = 0
    # Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)

        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, DepthwiseConv2D):
            config = layer.get_config()
            n_filters += config['filters']

        if isinstance(layer, DepthwiseConv2D):
            n_filters += layer.output_shape[-1]

    # Todo: Model contains Conv and Fully Connected layers
    # for layer_idx in range(1, len(model.get_layer(index=1))):
    #     layer = model.get_layer(index=1).get_layer(index=layer_idx)
    #     if isinstance(layer, keras.layers.Conv2D) == True:
    #         config = layer.get_config()
    #     n_filters += config['filters']
    return n_filters


def count_filters_layer(model):
    import keras
    # from keras.applications.mobilenet import DepthwiseConv2D
    from keras.layers import DepthwiseConv2D
    n_filters = ''
    # Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, DepthwiseConv2D):
            config = layer.get_config()
            n_filters += str(config['filters']) + ' '

        if isinstance(layer, DepthwiseConv2D):
            n_filters += str(layer.output_shape[-1])

    return n_filters


def compute_flops(model):
    # useful link https://www.programmersought.com/article/27982165768/
    import keras
    # from keras.applications.mobilenet import DepthwiseConv2D
    from keras.layers import DepthwiseConv2D
    total_flops = 0
    flops_per_layer = []

    for layer_idx in range(1, len(model.layers)):
        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, DepthwiseConv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

            # Computed according to https://arxiv.org/pdf/1704.04861.pdf Eq.(5)
            flops = (kernel_H * kernel_W * previous_layer_depth * output_map_H * output_map_W) + (
                    previous_layer_depth * current_layer_depth * output_map_W * output_map_H)
            total_flops += flops
            flops_per_layer.append(flops)

        elif isinstance(layer, keras.layers.Conv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

            flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
            total_flops += flops
            flops_per_layer.append(flops)

        if isinstance(layer, keras.layers.Dense) is True:
            _, current_layer_depth = layer.output_shape

            _, previous_layer_depth = layer.input_shape

            flops = current_layer_depth * previous_layer_depth
            total_flops += flops
            flops_per_layer.append(flops)

    return total_flops, flops_per_layer


def top_k_accuracy(y_true, y_pred, k):
    top_n = np.argsort(y_pred, axis=1)[:, -k:]
    idx_class = np.argmax(y_true, axis=1)
    hit = 0
    for i in range(idx_class.shape[0]):
        if idx_class[i] in top_n[i, :]:
            hit = hit + 1
    return float(hit) / idx_class.shape[0]


def center_crop(image, crop_size=224):
    h, w, _ = image.shape

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    bottom = top + crop_size
    right = left + crop_size

    return image[top:bottom, left:right, :]


def random_crop(img=None, random_crop_size=(64, 64)):
    # Code taken from https://jkjung-avt.github.io/keras-image-cropping/
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def data_augmentation(X, padding=4):
    X_out = np.zeros(X.shape, dtype=X.dtype)
    n_samples, x, y, _ = X.shape

    padded_sample = np.zeros((x + padding * 2, y + padding * 2, 3), dtype=X.dtype)

    for i in range(0, n_samples):
        p = random.random()
        padded_sample[padding:x + padding, padding:y + padding, :] = X[i][:, :, :]
        if p >= 0.5:  # random crop on the original image
            X_out[i] = random_crop(padded_sample, (x, y))
        else:  # random crop on the flipped image
            X_out[i] = random_crop(np.flip(padded_sample, axis=1), (x, y))

        # import matplotlib.pyplot as plt
        # plt.imshow(X_out[i])

    return X_out


def memory_usage(batch_size, model):
    import tensorflow as tf
    from keras import backend as K
    # Taken from #https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model

    if tf.__version__.split('.')[0] != '2':
        shapes_mem_count = 0
        for layer in model.layers:
            single_layer_mem = 1
            for s in layer.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    else:
        shapes_mem_count = 0
        for layer in model.layers:
            single_layer_mem = 1
            for s in layer.output_shape:
                if s is None or not isinstance(s, int):
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = total_memory / (1024.0 ** 3)
    return gbytes


def count_depth(model):
    import keras.layers as layers
    depth = 0
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, layers.Conv2D):
            depth = depth + 1
    print('Depth: [{}]'.format(depth))
    return depth
