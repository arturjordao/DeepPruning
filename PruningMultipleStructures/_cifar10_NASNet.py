import string
import sys

import keras
import numpy as np
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from sklearn.metrics._classification import accuracy_score
from tensorflow.python.data import Dataset

sys.path.insert(0, '../utils')
sys.path.insert(0, '../architectures')
sys.path.insert(0, 'PruningCriteria')
sys.path.insert(0, 'ResNet')

import custom_functions as func
import custom_callbacks

#################################################################################################################
"""NASNet-A models for Keras

NASNet refers to Neural Architecture Search Network, a family of models
that were designed automatically by learning the model architectures
directly on the dataset of interest.

Here we consider NASNet-A, the highest performance model that was found
for the CIFAR-10 dataset, and then extended to ImageNet 2012 dataset,
obtaining state of the art performance on CIFAR-10 and ImageNet 2012.
Only the NASNet-A models, and their respective weights, which are suited
for ImageNet 2012 are provided.

The below table describes the performance on ImageNet 2012:
------------------------------------------------------------------------------------
      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)
------------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.0 %  |   91.6 %  |       564 M    |     5.3        |
|   NASNet-A (6 @ 4032)  |   82.7 %  |   96.2 %  |      23.8 B    |    88.9        |
------------------------------------------------------------------------------------

Weights obtained from the official Tensorflow repository found at
https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet

# References:
 - [Learning Transferable Architectures for Scalable Image Recognition]
    (https://arxiv.org/abs/1707.07012)

Based on the following implementations:
 - [TF Slim Implementation]
   (https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.)
 - [TensorNets implementation]
   (https://github.com/taehoonlee/tensornets/blob/master/tensornets/nasnets.py)
"""

_BN_DECAY = 0.9997
_BN_EPSILON = 1e-3


def NASNet(input_shape=None,
           penultimate_filters=4032,
           nb_blocks=[6, 6, 6],
           stem_filters=96,
           initial_reduction=True,
           skip_reduction_layer_input=True,
           use_auxiliary_branch=False,
           filters_multiplier=2,
           dropout=0.5,
           weight_decay=5e-5,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000,
           default_size=None):
    """Instantiates a NASNet architecture.
    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(331, 331, 3)` for NASNetLarge or
            `(224, 224, 3)` for NASNetMobile
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        penultimate_filters: number of filters in the penultimate layer.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        nb_blocks: number of repeated blocks of the NASNet model.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        stem_filters: number of filters in the initial stem block
        skip_reduction: Whether to skip the reduction step at the tail
            end of the network. Set to `True` for CIFAR models.
        skip_reduction_layer_input: Determines whether to skip the reduction layers
            when calculating the previous layer to connect to.
        use_auxiliary_branch: Whether to use the auxiliary branch during
            training or evaluation.
        filters_multiplier: controls the width of the network.
            - If `filters_multiplier` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `filters_multiplier` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `filters_multiplier` = 1, default number of filters from the paper
                 are used at each layer.
        dropout: dropout rate
        weight_decay: l2 regularization weight
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: specifies the default image size of the model
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only Tensorflow backend is currently supported, '
                           'as other backends do not support '
                           'separable convolution.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    if default_size is None:
        default_size = 331

    if K.image_data_format() != 'channels_last':
        print('The NASNet family of models is only available '
              'for the input data format "channels_last" '
              '(width, height, channels). '
              'However your settings specify the default '
              'data format "channels_first" (channels, width, height).'
              ' You should set `image_data_format="channels_last"` '
              'in your Keras config located at ~/.keras/keras.json. '
              'The model being returned right now will expect inputs '
              'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    assert penultimate_filters % 24 == 0, "`penultimate_filters` needs to be divisible " \
                                          "by 24."

    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    filters = penultimate_filters // 24

    if initial_reduction:
        x = Conv2D(stem_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name='stem_conv1',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)
    else:
        x = Conv2D(stem_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='stem_conv1',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                           name='stem_bn1')(x)

    p = None
    if initial_reduction:  # imagenet / mobile mode
        x, p = _reduction_A(x, p, filters // (filters_multiplier ** 2), weight_decay, id='stem_1')
        x, p = _reduction_A(x, p, filters // filters_multiplier, weight_decay, id='stem_2')

    for i in range(nb_blocks[0]):
        x, p = _normal_A(x, p, filters, weight_decay, id='%d' % (i))

    x, p0 = _reduction_A(x, p, filters * filters_multiplier, weight_decay, id='reduce_%d' % (nb_blocks[0]))

    p = p0 if not skip_reduction_layer_input else p

    for i in range(nb_blocks[1]):
        x, p = _normal_A(x, p, filters * filters_multiplier, weight_decay, id='%d' % (nb_blocks[0] + i + 1))

    auxiliary_x = None
    if not initial_reduction:  # imagenet / mobile mode
        if use_auxiliary_branch:
            auxiliary_x = _add_auxiliary_head(x, classes, weight_decay, pooling, include_top)

    x, p0 = _reduction_A(x, p, filters * filters_multiplier ** 2, weight_decay,
                         id='reduce_%d' % (nb_blocks[0] + nb_blocks[1]))

    if initial_reduction:  # CIFAR mode
        if use_auxiliary_branch:
            auxiliary_x = _add_auxiliary_head(x, classes, weight_decay, pooling, include_top)

    p = p0 if not skip_reduction_layer_input else p

    for i in range(nb_blocks[2]):
        x, p = _normal_A(x, p, filters * filters_multiplier ** 2, weight_decay,
                         id='%d' % (nb_blocks[0] + nb_blocks[1] + i + 1))

    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout)(x)
        x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if use_auxiliary_branch:
        model = Model(inputs, [x, auxiliary_x], name='NASNet_with_auxiliary')
    else:
        model = Model(inputs, x, name='NASNet')

    # load weights
    if weights == 'imagenet':
        if default_size == 224:  # mobile version
            if include_top:
                if use_auxiliary_branch:
                    weight_path = NASNET_MOBILE_WEIGHT_PATH_WITH_AUXULARY
                    model_name = 'nasnet_mobile_with_aux.h5'
                else:
                    weight_path = NASNET_MOBILE_WEIGHT_PATH
                    model_name = 'nasnet_mobile.h5'
            else:
                if use_auxiliary_branch:
                    weight_path = NASNET_MOBILE_WEIGHT_PATH_WITH_AUXULARY_NO_TOP
                    model_name = 'nasnet_mobile_with_aux_no_top.h5'
                else:
                    weight_path = NASNET_MOBILE_WEIGHT_PATH_NO_TOP
                    model_name = 'nasnet_mobile_no_top.h5'

            weights_file = get_file(model_name, weight_path, cache_subdir='models')
            model.load_weights(weights_file)

        elif default_size == 331:  # large version
            if include_top:
                if use_auxiliary_branch:
                    weight_path = NASNET_LARGE_WEIGHT_PATH_WITH_auxiliary
                    model_name = 'nasnet_large_with_aux.h5'
                else:
                    weight_path = NASNET_LARGE_WEIGHT_PATH
                    model_name = 'nasnet_large.h5'
            else:
                if use_auxiliary_branch:
                    weight_path = NASNET_LARGE_WEIGHT_PATH_WITH_auxiliary_NO_TOP
                    model_name = 'nasnet_large_with_aux_no_top.h5'
                else:
                    weight_path = NASNET_LARGE_WEIGHT_PATH_NO_TOP
                    model_name = 'nasnet_large_no_top.h5'

            weights_file = get_file(model_name, weight_path, cache_subdir='models')
            model.load_weights(weights_file)

        else:
            raise ValueError('ImageNet weights can only be loaded on NASNetLarge or NASNetMobile')

    if old_data_format:
        K.set_image_data_format(old_data_format)

    return model


def NASNetCIFAR(input_shape=(32, 32, 3),
                dropout=0.0,
                weight_decay=5e-4,
                use_auxiliary_branch=False,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=10,
                depth_block=[6, 6, 6]):
    """Instantiates a NASNet architecture in CIFAR mode.
    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(32, 32, 3)` for NASNetMobile
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(32, 32, 3)` would be one valid value.
        use_auxiliary_branch: Whether to use the auxiliary branch during
            training or evaluation.
        dropout: dropout rate
        weight_decay: l2 regularization weight
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: specifies the default image size of the model
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    global _BN_DECAY, _BN_EPSILON
    _BN_DECAY = 0.9
    _BN_EPSILON = 1e-5

    return NASNet(input_shape,
                  penultimate_filters=768,
                  nb_blocks=depth_block,
                  stem_filters=32,
                  initial_reduction=False,
                  skip_reduction_layer_input=False,
                  use_auxiliary_branch=use_auxiliary_branch,
                  filters_multiplier=2,
                  dropout=dropout,
                  weight_decay=weight_decay,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  default_size=224)


def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1), weight_decay=5e-5, id=None):
    '''Adds 2 blocks of [relu-separable conv-batchnorm]

    # Arguments:
        ip: input tensor
        filters: number of output filters per layer
        kernel_size: kernel size of separable convolutions
        strides: strided convolution for downsampling
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        a Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('separable_conv_block_%s' % id):
        x = Activation('relu')(ip)
        x = SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % id,
                            padding='same', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name="separable_conv_1_bn_%s" % (id))(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % id,
                            padding='same', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name="separable_conv_2_bn_%s" % (id))(x)
    return x


def _adjust_block(p, ip, filters, weight_decay=5e-5, id=None):
    '''
    Adjusts the input `p` to match the shape of the `input`
    or situations where the output number of filters needs to
    be changed

    # Arguments:
        p: input tensor which needs to be modified
        ip: input tensor whose shape needs to be matched
        filters: number of output filters to be matched
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        an adjusted Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    img_dim = 2 if K.image_data_format() == 'channels_first' else -2

    with K.name_scope('adjust_block'):
        if p is None:
            p = ip

        elif p.shape[img_dim] != ip.shape[img_dim]:
            with K.name_scope('adjust_reduction_block_%s' % id):
                p = Activation('relu', name='adjust_relu_1_%s' % id)(p)

                p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_1_%s' % id)(p)
                p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                            name='adjust_conv_1_%s' % id, kernel_initializer='he_normal')(p1)

                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_2_%s' % id)(p2)
                p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                            name='adjust_conv_2_%s' % id, kernel_initializer='he_normal')(p2)

                p = concatenate([p1, p2], axis=channel_dim)
                p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                       name='adjust_bn_%s' % id)(p)

        elif p.shape[channel_dim] != filters:
            with K.name_scope('adjust_projection_block_%s' % id):
                p = Activation('relu')(p)
                p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='adjust_conv_projection_%s' % id,
                           use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(p)
                p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                       name='adjust_bn_%s' % id)(p)
    return p


def _normal_A(ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Normal cell for NASNet-A (Fig. 4 in the paper)

    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        a Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('normal_A_block_%s' % id):
        p = _adjust_block(p, ip, filters, weight_decay, id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % id,
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name='normal_bn_1_%s' % id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5), weight_decay=weight_decay,
                                         id='normal_left1_%s' % id)
            x1_2 = _separable_conv_block(p, filters, weight_decay=weight_decay, id='normal_right1_%s' % id)
            x1 = add([x1_1, x1_2], name='normal_add_1_%s' % id)

        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5, 5), weight_decay=weight_decay, id='normal_left2_%s' % id)
            x2_2 = _separable_conv_block(p, filters, (3, 3), weight_decay=weight_decay, id='normal_right2_%s' % id)
            x2 = add([x2_1, x2_2], name='normal_add_2_%s' % id)

        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left3_%s' % (id))(h)
            x3 = add([x3, p], name='normal_add_3_%s' % id)

        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % (id))(p)
            x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_right4_%s' % (id))(p)
            x4 = add([x4_1, x4_2], name='normal_add_4_%s' % id)

        with K.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, weight_decay=weight_decay, id='normal_left5_%s' % id)
            x5 = add([x5, h], name='normal_add_5_%s' % id)

        x = concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name='normal_concat_%s' % id)
    return x, ip


def _reduction_A(ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Reduction cell for NASNet-A (Fig. 4 in the paper)

    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        a Keras tensor
    '''
    """"""
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('reduction_A_block_%s' % id):
        p = _adjust_block(p, ip, filters, weight_decay, id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % id,
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name='reduction_bn_1_%s' % id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_left1_%s' % id)
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_1_%s' % id)
            x1 = add([x1_1, x1_2], name='reduction_add_1_%s' % id)

        with K.name_scope('block_2'):
            x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left2_%s' % id)(h)
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_right2_%s' % id)
            x2 = add([x2_1, x2_2], name='reduction_add_2_%s' % id)

        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left3_%s' % id)(h)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_right3_%s' % id)
            x3 = add([x3_1, x3_2], name='reduction_add3_%s' % id)

        with K.name_scope('block_4'):
            x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='reduction_left4_%s' % id)(x1)
            x4 = add([x2, x4])

        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3, 3), weight_decay=weight_decay, id='reduction_left4_%s' % id)
            x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_right5_%s' % id)(h)
            x5 = add([x5_1, x5_2], name='reduction_add4_%s' % id)

        x = concatenate([x2, x3, x4, x5], axis=channel_dim, name='reduction_concat_%s' % id)
        return x, ip


def _add_auxiliary_head(x, classes, weight_decay, pooling, include_top):
    '''Adds an auxiliary head for training the model

    From section A.7 "Training of ImageNet models" of the paper, all NASNet models are
    trained using an auxiliary classifier around 2/3 of the depth of the network, with
    a loss weight of 0.4

    # Arguments
        x: input tensor
        classes: number of output classes
        weight_decay: l2 regularization weight

    # Returns
        a keras Tensor
    '''
    img_height = 1 if K.image_data_format() == 'channels_last' else 2
    img_width = 2 if K.image_data_format() == 'channels_last' else 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('auxiliary_branch'):
        auxiliary_x = Activation('relu')(x)
        auxiliary_x = AveragePooling2D((5, 5), strides=(3, 3), padding='valid', name='aux_pool')(auxiliary_x)
        auxiliary_x = Conv2D(128, (1, 1), padding='same', use_bias=False, name='aux_conv_projection',
                             kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(auxiliary_x)
        auxiliary_x = BatchNormalization(axis=channel_axis, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                         name='aux_bn_projection')(auxiliary_x)
        auxiliary_x = Activation('relu')(auxiliary_x)

        auxiliary_x = Conv2D(768, (auxiliary_x.shape[img_height], auxiliary_x.shape[img_width]),
                             padding='valid', use_bias=False, kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_decay), name='aux_conv_reduction')(auxiliary_x)
        auxiliary_x = BatchNormalization(axis=channel_axis, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                         name='aux_bn_reduction')(auxiliary_x)
        auxiliary_x = Activation('relu')(auxiliary_x)

        if include_top:
            auxiliary_x = Flatten()(auxiliary_x)
            auxiliary_x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay),
                                name='aux_predictions')(auxiliary_x)
        else:
            if pooling == 'avg':
                auxiliary_x = GlobalAveragePooling2D()(auxiliary_x)
            elif pooling == 'max':
                auxiliary_x = GlobalMaxPooling2D()(auxiliary_x)

    return auxiliary_x


#################################################################################################################

def random_idx(model, blocks, p_layers=0.1):
    num_blocks = blocks
    allowed_layers = concat_to_prune(model)
    n_layers = len(allowed_layers)
    num_remove = int(p_layers * n_layers)
    mask = np.ones(len(allowed_layers))
    score_block = idx_score_block(model, allowed_layers)

    scores = np.arange(len(allowed_layers)) * 1.0
    np.random.shuffle(scores)

    for i in range(0, num_remove):
        min_vip = np.argmin(scores)
        block_idx = allowed_layers[min_vip]
        block_idx = score_block[block_idx]
        if num_blocks[block_idx] - 1 > 1:
            mask[min_vip] = 0
            num_blocks[block_idx] = num_blocks[block_idx] - 1
        else:
            print('Warning: It does not possible to remove more layer from block [{}]'.format(block_idx))

        scores[min_vip] = np.inf  # Removes the minimum VIP from the list
        # print('Removing Concat [{}]'.format(allowed_layers[min_vip]), flush=True)

    return num_blocks, mask


def statistics(model):
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    flops, _ = func.compute_flops(model)
    blocks = count_nas_blocks(model)

    memory = func.memory_usage(1, model)
    print('Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
          'Memory [{:.6f}]'.format(blocks, n_params, n_filters, flops, memory), flush=True)


def concat_to_prune(model):
    allowed_layers = []
    all_concat = []

    for i in range(0, len(model.layers)):
        if model.get_layer(index=i).name.__contains__('normal_concat_'):
            all_concat.append(i)

    # The first tow concat cannot be removed
    # We can start removing from adjust_conv_projection_2
    all_concat.pop(0)
    all_concat.pop(0)

    for i in range(0, len(all_concat)):
        name = model.get_layer(index=all_concat[i] + 3).name
        if not name.__contains__('reduction_conv'):

            name = model.get_layer(index=all_concat[i] - 42).name
            if name.__contains__('adjust_conv_projection_'):

                name = model.get_layer(index=all_concat[i] - 92).name
                if not name.__contains__('adjust_avg_pool'):
                    # These are the valid blocks we can remove
                    allowed_layers.append(all_concat[i])

    # # The last block is enabled
    # allowed_layers.append(all_concat[-1])
    return allowed_layers


def print_debug(model):
    for i in range(0, len(model.layers)):
        print('Index [{}] {} {}'.format(i, model.get_layer(index=i).name, model.get_layer(index=i).output_shape))


def reduce_layers(model):
    allowed_layers = []

    for i in range(0, len(model.layers)):
        if model.get_layer(index=i).name.__contains__('reduction_conv_1_reduce'):
            allowed_layers.append(np.arange(i - 45, i + 47))

        if model.get_layer(index=i).name.__contains__('reduction_add3_reduce_'):
            allowed_layers.append(np.arange(i, i + 54))

        if model.get_layer(index=i).name.__contains__('reduction_add3_reduce_'):
            if model.get_layer(index=i).output_shape == (None, 16, 16, 64):
                idx = i
                while not model.get_layer(index=idx).name.__contains__('adjust_conv_projection_'):
                    idx = idx + 1
                allowed_layers.append(np.arange(idx, idx + 45))

        if model.get_layer(index=i).name.__contains__('normal_concat_'):
            if model.get_layer(index=i).output_shape == (None, 8, 8, 768):
                if model.get_layer(index=i - 44).output_shape == (None, 8, 8, 512):
                    allowed_layers.append(np.arange(i - 42, i + 1))

    allowed_layers = [item for sublist in allowed_layers for item in sublist]
    return allowed_layers


def transfer_weights(model, new_model, mask):
    # TODO: tem que copiar todos os pesos ate chegar em adjust_conv_projection_2
    # So a partir de adjust_conv_projection_2 eh que os blocos podem ser removidos

    assigned_weights = np.zeros((len(new_model.layers)), dtype=bool)
    for i in range(0, 88):
        w = model.get_layer(index=i).get_weights()
        new_model.get_layer(index=i).set_weights(w)
        assigned_weights[i] = True

    # These are the reduce layers
    # Here it is necessary match by name and index -- Keras problems
    idx_model = reduce_layers(model)
    idx_new_model = reduce_layers(new_model)

    lname_model = []
    for i in idx_model:
        name = model.get_layer(index=i).name.rstrip(string.digits)
        shape = str(model.get_layer(index=i).output_shape) + str(model.get_layer(index=i).input_shape)
        lname_model.append(name + shape)  # This ID avoid conflicts between blocks
        # E.g., reduction_conv_1_reduce_ from block 1 and 2

    lname_new_model = []
    for i in idx_new_model:
        name = new_model.get_layer(index=i).name.rstrip(string.digits)
        shape = str(new_model.get_layer(index=i).output_shape) + str(new_model.get_layer(index=i).input_shape)
        lname_new_model.append(name + shape)  # This ID avoid conflicts between blocks
        # E.g., reduction_conv_1_reduce_ from block 1 and 2

    for transfer_idx in range(0, len(idx_new_model)):
        name_id = lname_new_model[transfer_idx]

        if name_id in lname_model:
            # print('{} {}'.format(idx_model[lname_model.index(name_id)], idx_new_model[transfer_idx]))
            w = model.get_layer(index=idx_model[lname_model.index(name_id)]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)
            assigned_weights[idx_new_model[transfer_idx]] = True

            # A = model.get_layer(index=idx_model[lname_model.index(name_id)]).name
            # B = new_model.get_layer(index=idx_new_model[transfer_idx]).name
            # print('{}            {}'.format(A, B))

            # Once the layer provide weights we need to remove it from list
            # To avoid that it provides weights twice (due to same input/output shape)
            lname_model[lname_model.index(name_id)] = ''  # We cannot use pop() function

    # These are the layers where the weights must to be transfered
    concat_model = concat_to_prune(model)
    concat_new_model = concat_to_prune(new_model)

    concat_model = np.array(concat_model)[mask == 1]
    concat_model = list(concat_model)
    end = len(concat_new_model)

    for layer_idx in range(0, end):

        idx_model = np.arange(concat_model[0] - 42, concat_model[0] + 1).tolist()
        idx_new_model = np.arange(concat_new_model[0] - 42, concat_new_model[0] + 1).tolist()

        for transfer_idx in range(0, len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)
            assigned_weights[idx_new_model[transfer_idx]] = True

            # A = model.get_layer(index=idx_model[transfer_idx]).name
            # B = new_model.get_layer(index=idx_new_model[transfer_idx]).name
            # print('{}            {}'.format(A, B))

        concat_new_model.pop(0)
        concat_model.pop(0)

    # This is the classification layer
    w = model.get_layer(index=-1).get_weights()
    new_model.get_layer(index=-1).set_weights(w)
    assigned_weights[-1] = True

    for i in range(0, len(assigned_weights)):
        if assigned_weights[i] == False:
            if not isinstance(new_model.get_layer(index=i), Activation) and not isinstance(new_model.get_layer(index=i),
                                                                                           Add):
                if not isinstance(new_model.get_layer(index=i), GlobalAveragePooling2D) and not isinstance(
                        new_model.get_layer(index=i), Dropout):
                    if not isinstance(new_model.get_layer(index=i), Concatenate):
                        print('Weights from Layer[{}] were not transferred'.format(i))

    return new_model


def idx_score_block(model, layers):
    # Associates the layer index with the NASNet block
    output = {}
    shapes = [32, 16, 8]  # Shapes of blocks 0, 1, 2

    for i in range(0, len(layers)):
        shape = model.get_layer(index=layers[i]).output_shape[1]
        idx = shapes.index(shape)
        output[layers[i]] = idx

    return output


def count_nas_blocks(model):
    concat_layers = []
    shapes = [32, 16, 8]  # Shapes of blocks 0, 1, 2

    for i in range(0, len(model.layers)):
        if model.get_layer(index=i).name.__contains__('normal_concat'):
            concat_layers.append(i)

    blocks = np.zeros((len(shapes)), dtype=int)

    for i in range(0, len(concat_layers)):
        shape = model.get_layer(index=concat_layers[i]).output_shape[1]
        idx = shapes.index(shape)
        blocks[idx] = blocks[idx] + 1

    return blocks


def finetuning(model, X_train, y_train, X_test, y_test):
    lr = 0.01
    schedule = [(100, lr / 10), (150, lr / 100)]
    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
    print('Accuracy before fine-tuning  [{:.4f}]'.format(acc), flush=True)

    for ep in range(0, 200):
        y_tmp = np.concatenate((y_train, y_train, y_train))
        X_tmp = np.concatenate(
            (func.data_augmentation(X_train),
             func.data_augmentation(X_train),
             func.data_augmentation(X_train)))

        # with tf.device("CPU"):
        X_tmp = Dataset.from_tensor_slices((X_tmp, y_tmp)).shuffle(4 * 128).batch(128)

        model.fit(X_tmp, batch_size=128,
                  callbacks=callbacks, verbose=2,
                  epochs=ep, initial_epoch=ep - 1)

        if ep % 5 == 0:  # % 5
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
            print('Accuracy [{:.4f}]'.format(acc), flush=True)

    return model


if __name__ == '__main__':
    np.random.seed(12227)

    p = 0.4

    depth = 241  # ResNet types: 32, 56, 110
    architecture_name = 'NASNet{}'.format(depth)  # 'ResNet56', 'ResNet110',

    method = 'PLS+VIP'  # PLS+VIP, infFS, ilFS
    n_components = 2

    print('Architecture [{}] Method[{}] #Components[{}] Pruned[{}]'.format(architecture_name, method, n_components, p),
          flush=True)

    lr = 0.01
    schedule = [(100, 1e-3), (150, 1e-4)]

    model = func.load_model(architecture_file='../architectures/CIFAR10/{}'.format(architecture_name),
                            weights_file='G:/Meu Drive/Projects/weights/CIFAR10/{}++'.format(architecture_name))
    # model = NASNetCIFAR((32, 32, 3), depth_block=[6, 6, 6], classes=10)
    X_train, y_train, X_test, y_test = func.cifar_resnet_data(debug=False)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
    print('Accuracy [{:.4f}]'.format(acc), flush=True)
    statistics(model)

    # blocks = count_nas_blocks(model)
    # allowed_layers = concat_to_prune(model)

    for i in range(0, 50):
        blocks, mask = random_idx(model, count_nas_blocks(model), p)
        tmp_model = NASNetCIFAR((32, 32, 3), depth_block=blocks, classes=10)
        model = transfer_weights(model, tmp_model, mask)

        print('Results before fine-tuning', flush=True)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
        print('Accuracy [{:.4f}]'.format(acc), flush=True)

        # model = fine_tuning(model)

        statistics(model)
        y_pred = model.predict(X_test)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print('Accuracy [{:.4f}]'.format(acc), flush=True)

        # func.save_model('CIFAR10/'+architecture_name + '_' + method + '_Blocks{}_p[{}]_c[{}]'.format(blocks, p, n_components), model)
