# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
from keras import backend
from keras import layers
from keras import models
from keras.layers import *


# from keras.applications.imagenet_utils import _obtain_input_shape

def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def resnet(input_shape=(224, 224, 3), blocks=None, num_classes=1000):
    stacks = len(blocks)
    num_filters = 64
    bn_axis = 3

    img_input = Input(input_shape)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = Conv2D(num_filters, 7, strides=2, use_bias=True, name='conv1_conv')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    for stage in range(0, stacks):
        num_res_blocks = blocks[stage]
        name = 'conv' + str(stage + 2)
        if stage == 0:
            stride = 1
        else:
            stride = 2
        x = block1(x, filters=num_filters, stride=stride, name=name + '_block1')

        for res_block in range(2, num_res_blocks + 1):
            x = block1(x, num_filters, conv_shortcut=False, name=name + '_block' + str(res_block))

        num_filters = num_filters * 2

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation='softmax', name='probs')(x)
    inputs = img_input

    model = models.Model(inputs, x, name='ResNetBN')
    return model


if __name__ == '__main__':
    import numpy as np
    import sys
    import h5py

    sys.path.insert(0, '../utils')

    import custom_functions as func
    from keras.applications.resnet import preprocess_input
    from sklearn.utils import gen_batches
    from sklearn.metrics import accuracy_score

    np.random.seed(12227)

    input_shape = (224, 224, 3)
    num_classes = 1000

    blocks = [3, 4, 6, 3]
    # x = stack1(x, 64, 3, stride1=1, name='conv2')
    # x = stack1(x, 128, 4, name='conv3')
    # x = stack1(x, 256, 6, name='conv4')
    # x = stack1(x, 512, 3, name='conv5')
    model = resnet(input_shape=input_shape,
                   blocks=blocks,
                   num_classes=num_classes)

    # Absolute paths :(
    model.load_weights('E:/Projects/weights/resnet_common/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    tmp = h5py.File('E:/datasets/ImageNet/processed_data/imageNet_images.h5', 'r')

    X_test, y_test = tmp['X_test'], tmp['y_test']

    y_pred = np.zeros((X_test.shape[0], y_test.shape[1]))
    for batch in gen_batches(X_test.shape[0], 1024):
        samples = preprocess_input(X_test[batch].astype(float))
        y_pred[batch] = model.predict(samples, batch_size=256)

    top1_error = 1 - accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    top5_error = 1 - func.top_k_accuracy(y_test, y_pred, 5)
    top10_error = 1 - func.top_k_accuracy(y_test, y_pred, 10)

    print('Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.format(top1_error, top5_error, top10_error))
