import numpy as np
from keras import backend as K
from keras.layers import *
from keras.models import Model

import rebuild_layers as rl  # It implements some particular functions we need to use here


# Used by MobileNet
def relu6(x):
    return K.relu(x, max_value=6)


def rw_bn(w, index):
    w[0] = np.delete(w[0], index)
    w[1] = np.delete(w[1], index)
    w[2] = np.delete(w[2], index)
    w[3] = np.delete(w[3], index)
    return w


def rw_cn(index_model, idx_pruned, model):
    # This function removes the weights of the Conv2D considering the previous prunning in other Conv2D
    config = model.get_layer(index=index_model).get_config()
    weights = model.get_layer(index=index_model).get_weights()
    weights[0] = np.delete(weights[0], idx_pruned, axis=2)
    return create_Conv2D_from_conf(config, weights)


def create_Conv2D_from_conf(config, weights):
    n_filters = weights[0].shape[-1]
    return Conv2D(activation=config['activation'],
                  activity_regularizer=config['activity_regularizer'],
                  bias_constraint=config['bias_constraint'],
                  bias_regularizer=config['bias_regularizer'],
                  data_format=config['data_format'],
                  dilation_rate=config['dilation_rate'],
                  filters=n_filters,
                  kernel_constraint=config['kernel_constraint'],
                  kernel_regularizer=config['kernel_regularizer'],
                  kernel_size=config['kernel_size'],
                  name=config['name'],
                  padding=config['padding'],
                  strides=config['strides'],
                  trainable=config['trainable'],
                  use_bias=config['use_bias'],
                  weights=weights
                  )


def create_depthwise_from_config(config, weights):
    return DepthwiseConv2D(activation=config['activation'],
                           activity_regularizer=config['activity_regularizer'],
                           bias_constraint=config['bias_constraint'],
                           bias_regularizer=config['bias_regularizer'],
                           data_format=config['data_format'],
                           dilation_rate=config['dilation_rate'],
                           depth_multiplier=config['depth_multiplier'],
                           depthwise_constraint=config['depthwise_constraint'],
                           depthwise_initializer=config['depthwise_initializer'],
                           depthwise_regularizer=config['depthwise_regularizer'],
                           kernel_size=config['kernel_size'],
                           name=config['name'],
                           padding=config['padding'],
                           strides=config['strides'],
                           trainable=config['trainable'],
                           use_bias=config['use_bias'],
                           weights=weights
                           )


def remove_conv_weights(index_model, idxs, model):
    config, weights = (model.get_layer(index=index_model).get_config(),
                       model.get_layer(index=index_model).get_weights())
    weights[0] = np.delete(weights[0], idxs, axis=3)
    weights[1] = np.delete(weights[1], idxs)
    config['filters'] = weights[1].shape[0]
    return idxs, config, weights


def remove_convMobile_weights(index_model, idxs, model):
    config, weights = (model.get_layer(index=index_model).get_config(),
                       model.get_layer(index=index_model).get_weights())
    weights[0] = np.delete(weights[0], idxs, axis=3)
    config['filters'] = weights[0].shape[-1]
    return idxs, config, weights


def rebuild_resnet(model, blocks, layer_filters, num_classes=10):
    num_filters = 16
    num_res_blocks = blocks

    inputs = Input(shape=(model.inputs[0].shape.dims[1].value,
                          model.inputs[0].shape.dims[2].value,
                          model.inputs[0].shape.dims[3].value))

    # The first bock is not allow to prune
    _, config, weights = remove_conv_weights(1, [], model)
    conv = create_Conv2D_from_conf(config, weights)

    H = conv(inputs)
    H = BatchNormalization(weights=model.get_layer(index=2).get_weights())(H)
    H = Activation.from_config(model.get_layer(index=3).get_config())(H)
    x = H

    i = 4

    remove_Conv2D = [item[1] for item in layer_filters]
    remove_Conv2D.reverse()
    layer_block = False
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks[stack]):

            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            # This is the layer we can prune
            idx_previous, config, weights = remove_conv_weights(i, remove_Conv2D.pop(), model)
            conv = create_Conv2D_from_conf(config, weights)
            i = i + 1
            y = conv(x)
            wb = model.get_layer(index=i).get_weights()
            y = BatchNormalization(weights=rw_bn(wb, idx_previous))(y)
            i = i + 1
            y = Activation.from_config(model.get_layer(index=i).get_config())(y)
            i = i + 1

            # Second Module
            conv = rw_cn(index_model=i, idx_pruned=idx_previous, model=model)
            i = i + 1
            y = conv(y)  # Aqui embaixo vai ter que ter um if relacionado ao bloco
            if layer_block == False:
                y = BatchNormalization(weights=model.get_layer(index=i).get_weights())(y)
            else:
                y = BatchNormalization(weights=model.get_layer(index=i + 1).get_weights())(y)
                layer_block = False
            i = i + 1

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                _, config, weights = remove_conv_weights(i - 1, [], model)
                conv = create_Conv2D_from_conf(config, weights)
                x = conv(x)
                i = i + 1

            x = Add()([x, y])
            i = i + 1
            # x = Activation('relu')(x)
            x = Activation.from_config(model.get_layer(index=i).get_config())(x)
            i = i + 1
        num_filters *= 2
        layer_block = True

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    layer = model.get_layer(index=-1)
    config = layer.get_config()
    weights = layer.get_weights()
    outputs = Dense(units=config['units'],
                    activation=config['activation'],
                    activity_regularizer=config['activity_regularizer'],
                    bias_constraint=config['bias_constraint'],
                    bias_regularizer=config['bias_regularizer'],
                    kernel_constraint=config['kernel_constraint'],
                    kernel_regularizer=config['kernel_regularizer'],
                    name=config['name'],
                    trainable=config['trainable'],
                    use_bias=config['use_bias'],
                    weights=weights)(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def rebuild_resnetBN(model, blocks, layer_filters, iter=0, num_classes=1000):
    stacks = len(blocks)
    num_filters = 64
    layer_filters = dict(layer_filters)

    inputs = Input(shape=(model.inputs[0].shape.dims[1].value,
                          model.inputs[0].shape.dims[2].value,
                          model.inputs[0].shape.dims[3].value))

    # ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = ZeroPadding2D.from_config(config=model.get_layer(index=1).get_config())(inputs)

    # Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
    _, config, weights = remove_conv_weights(2, [], model)
    conv = create_Conv2D_from_conf(config, weights)
    x = conv(x)

    # x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
    #                        name='conv1_bn')(x)
    x = BatchNormalization(name='BN00' + str(iter), weights=model.get_layer(index=3).get_weights(), epsilon=1.001e-5)(x)

    # x = Activation('relu', name='conv1_relu')(x)
    x = Activation.from_config(config=model.get_layer(index=4).get_config())(x)

    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = ZeroPadding2D.from_config(config=model.get_layer(index=5).get_config())(x)

    # x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
    x = MaxPooling2D.from_config(config=model.get_layer(index=6).get_config())(x)

    i = 7
    for stage in range(0, stacks):
        num_res_blocks = blocks[stage]

        # First Layer Block
        # x = block1(x, filters=num_filters, stride=stride, name=name + '_block1')
        shortcut = x

        # x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)
        i = i + 1

        # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
        x = BatchNormalization(name='BN' + str(i) + str(iter), weights=model.get_layer(index=i).get_weights(),
                               epsilon=1.001e-5)(x)
        i = i + 1

        # x = layers.Activation('relu', name=name + '_1_relu')(x)
        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        # x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)
        i = i + 1

        # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
        x = BatchNormalization(name='BN' + str(i) + str(iter), weights=model.get_layer(index=i).get_weights(),
                               epsilon=1.001e-5)(x)
        i = i + 1

        # x = layers.Activation('relu', name=name + '_2_relu')(x)
        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        _, config, weights = remove_conv_weights(i + 1, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)

        x = BatchNormalization(name='BN' + str(i) + str(iter), weights=model.get_layer(index=i + 3).get_weights(),
                               epsilon=1.001e-5)(x)

        # x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        shortcut = conv(shortcut)
        i = i + 2

        # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_3_bn')(x)
        shortcut = BatchNormalization(name='BN' + str(i) + str(iter), weights=model.get_layer(index=i).get_weights(),
                                      epsilon=1.001e-5)(shortcut)
        i = i + 1

        # x = layers.Add(name=name + '_add')([shortcut, x])
        x = Add(name=model.get_layer(index=i).name)([shortcut, x])
        i = i + 2

        # x = layers.Activation('relu', name=name + '_out')(x)
        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        # end First Layer Block

        for res_block in range(2, num_res_blocks + 1):
            # x = block1(x, num_filters, conv_shortcut=False, name=name + '_block' + str(res_block))
            shortcut = x

            # x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
            idx_previous, config, weights = remove_conv_weights(i, layer_filters.get(i, []), model)
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
            wb = model.get_layer(index=i).get_weights()
            x = BatchNormalization(name='BN' + str(i) + str(iter), weights=rw_bn(wb, idx_previous), epsilon=1.001e-5)(x)
            i = i + 1

            # x = layers.Activation('relu', name=name + '_1_relu')(x)
            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

            # x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
            weights = model.get_layer(index=i).get_weights()
            config = model.get_layer(index=i).get_config()
            idxs = layer_filters.get(i, [])
            weights[0] = np.delete(weights[0], idxs, axis=3)
            weights[1] = np.delete(weights[1], idxs)
            weights[0] = np.delete(weights[0], idx_previous, axis=2)
            idx_previous = idxs
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
            wb = model.get_layer(index=i).get_weights()
            x = BatchNormalization(name='BN' + str(i) + str(iter), weights=rw_bn(wb, idx_previous), epsilon=1.001e-5)(x)
            i = i + 1

            # x = layers.Activation('relu', name=name + '_2_relu')(x)
            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

            # x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
            weights = model.get_layer(index=i).get_weights()
            config = model.get_layer(index=i).get_config()
            weights[0] = np.delete(weights[0], idx_previous, axis=2)
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)
            x = BatchNormalization(name='BN' + str(i) + str(iter), weights=model.get_layer(index=i).get_weights(),
                                   epsilon=1.001e-5)(x)
            i = i + 1

            # x = layers.Add(name=name + '_add')([shortcut, x])
            x = Add.from_config(config=model.get_layer(index=i).get_config())([shortcut, x])
            i = i + 1

            # x = layers.Activation('relu', name=name + '_out')(x)
            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

        num_filters = num_filters * 2

    # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = GlobalAveragePooling2D.from_config(config=model.get_layer(index=i).get_config())(x)
    i = i + 1

    # x = layers.Dense(num_classes, activation='softmax', name='probs')(x)
    weights = model.get_layer(index=i).get_weights()
    config = model.get_layer(index=i).get_config()
    x = Dense(units=config['units'], activation=config['activation'], weights=weights)(x)

    model = Model(inputs, x, name='ResNetBN')
    return model


def rebuild_mobilenetV2(model, blocks, layer_filters, initial_reduction=False, num_classes=1000):
    blocks = np.append(blocks, 1)
    stacks = len(blocks)
    layer_filters = dict(layer_filters)

    inputs = Input(shape=(model.inputs[0].shape.dims[1].value,
                          model.inputs[0].shape.dims[2].value,
                          model.inputs[0].shape.dims[3].value))

    idx_previous = []
    i = 1
    if isinstance(model.get_layer(index=i), ZeroPadding2D):
        x = ZeroPadding2D.from_config(model.get_layer(index=i).get_config())(inputs)  # model.get_layer(index=i)(inputs)
        i = i + 1

        config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
        x = create_Conv2D_from_conf(config, weights)(x)  # model.get_layer(index=i)(x)
        i = i + 1

        x = BatchNormalization(name=model.get_layer(index=i).name,
                               weights=model.get_layer(index=i).get_weights(),
                               epsilon=1e-3, momentum=0.999)(x)  # model.get_layer(index=i)(x)
        i = i + 1

        x = Activation(relu6, name=model.get_layer(index=i).name)(x)  # model.get_layer(index=i)(x)
        i = i + 1

    else:
        x = inputs

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    x = create_depthwise_from_config(config, weights)(x)  # model.get_layer(index=i)(x)
    i = i + 1

    x = BatchNormalization(name=model.get_layer(index=i).name,
                           weights=model.get_layer(index=i).get_weights(),
                           epsilon=1e-3, momentum=0.999)(x)  # model.get_layer(index=i)(x)
    i = i + 1

    x = Activation(relu6, name=model.get_layer(index=i).name)(x)  # model.get_layer(index=i)(x)
    i = i + 1

    idx_previous, config, weights = remove_convMobile_weights(i, layer_filters.get(i, []), model)
    x = create_Conv2D_from_conf(config, weights)(x)
    i = i + 1

    wb = model.get_layer(index=i).get_weights()
    x = BatchNormalization(name=model.get_layer(index=i).name,
                           weights=rw_bn(wb, idx_previous),
                           epsilon=1e-3, momentum=0.999)(x)
    i = i + 1

    id = 1
    for stage in range(0, stacks):
        num_blocks = blocks[stage]

        for mobile_block in range(0, num_blocks):
            prefix = 'block_{}_'.format(id)
            # print(prefix)
            shortcut = x

            # 1x1 Convolution -- _expand
            idx = layer_filters.get(i, [])
            config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
            weights[0] = np.delete(weights[0], idx, axis=3)

            # First block expand only
            if id == 1:
                weights[0] = np.delete(weights[0], idx_previous, axis=2)

            x = create_Conv2D_from_conf(config, weights)(x)
            # idx_previous = idx
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            x = BatchNormalization(name=prefix + 'expand_BN', weights=rw_bn(wb, idx),
                                   epsilon=1e-3, momentum=0.999)(x)
            i = i + 1

            x = Activation(relu6, name=model.get_layer(index=i).name)(x)  # model.get_layer(index=i)(x)
            i = i + 1

            if isinstance(model.get_layer(index=i), ZeroPadding2D):  # stride==2
                x = ZeroPadding2D.from_config(model.get_layer(index=i).get_config())(x)  # model.get_layer(index=i)(x)
                i = i + 1

            # block_kth_depthwise
            config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
            weights[0] = np.delete(weights[0], idx, axis=2)
            x = create_depthwise_from_config(config, weights)(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            x = BatchNormalization(name=model.get_layer(index=i).name,
                                   epsilon=1e-3, momentum=0.999, weights=rw_bn(wb, idx))(x)
            i = i + 1

            x = Activation(relu6, name=model.get_layer(index=i).name)(x)  # model.get_layer(index=i)(x)
            i = i + 1

            # block_kth_project
            x = rw_cn(i, idx, model)(x)
            i = i + 1

            x = BatchNormalization(name=model.get_layer(index=i).name,
                                   weights=model.get_layer(index=i).get_weights(),
                                   epsilon=1e-3, momentum=0.999)(x)  # model.get_layer(index=i)(x)
            i = i + 1

            if isinstance(model.get_layer(index=i), Add):
                x = Add(name=model.get_layer(index=i).name)([shortcut, x])
                i = i + 1

            id = id + 1

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    x = create_Conv2D_from_conf(config, weights)(x)  # model.get_layer(index=i+1)(x)

    x = BatchNormalization(name=model.get_layer(index=i + 1).name,
                           weights=model.get_layer(index=i + 1).get_weights(),
                           epsilon=1e-3, momentum=0.999)(x)

    x = Activation(relu6, name=model.get_layer(index=i + 2).name)(x)  # model.get_layer(index=i+2)(x)

    x = GlobalAveragePooling2D.from_config(model.get_layer(index=i + 3).get_config())(
        x)  # model.get_layer(index=i+3)(x)

    config, weights = model.get_layer(index=i + 4).get_config(), model.get_layer(index=i + 4).get_weights()
    x = Dense(units=config['units'],
              activation=config['activation'],
              activity_regularizer=config['activity_regularizer'],
              bias_constraint=config['bias_constraint'],
              bias_regularizer=config['bias_regularizer'],
              kernel_constraint=config['kernel_constraint'],
              kernel_regularizer=config['kernel_regularizer'],
              name=config['name'],
              trainable=config['trainable'],
              use_bias=config['use_bias'],
              weights=weights)(x)  # model.get_layer(index=i+4)(x)

    model = Model(inputs, x, name='MobileNetV2')
    return model


def allowed_layers_resnet(model):
    allowed_layers = []
    all_add = []
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, Add):
            all_add.append(i)
        if isinstance(layer, Conv2D) and layer.strides == (2, 2) and layer.kernel_size != (1, 1):
            allowed_layers.append(i)

    allowed_layers.append(all_add[0] - 5)

    for i in range(1, len(all_add)):
        allowed_layers.append(all_add[i] - 5)

    # To avoid bug due to keras architecture (i.e., order of layers)
    # This ensure that only Conv2D are "allowed layers"
    tmp = allowed_layers
    allowed_layers = []

    for i in tmp:
        if isinstance(model.get_layer(index=i), Conv2D):
            allowed_layers.append(i)

    # allowed_layers.append(all_add[-1] - 5)
    return allowed_layers


def allowed_layers_resnetBN(model):
    allowed_layers = []
    all_add = []
    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output_shape
        output_shape = model.get_layer(index=all_add[i - 1]).output_shape
        # These are the valid blocks we can remove
        if input_shape == output_shape:
            allowed_layers.append(all_add[i] - 8)
            allowed_layers.append(all_add[i] - 5)

    return allowed_layers


def allowed_layers_mobilenetV2(model):
    allowed_layers = []
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)

        if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            if layer.name.__contains__('expand'):
                allowed_layers.append(i)

    return allowed_layers


def idx_to_conv2Didx(model, indices):
    # Convert index onto Conv2D index (required by pruning methods)
    idx_Conv2D = 0
    output = []
    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Conv2D):
            if i in indices:
                output.append(idx_Conv2D)

            idx_Conv2D = idx_Conv2D + 1

    return output


def layer_to_prune_filters(model):
    if architecture_name.__contains__('ResNet'):
        if architecture_name.__contains__('50'):  # ImageNet archicttures (ResNet50, 101 and 152)
            allowed_layers = allowed_layers_resnetBN(model)
        else:  # CIFAR-like archictures (low-resolution datasets)
            allowed_layers = allowed_layers_resnet(model)

    if architecture_name.__contains__('MobileNetV2'):
        allowed_layers = allowed_layers_mobilenetV2(model)

    # allowed_layers = idx_to_conv2Didx(model, allowed_layers)
    return allowed_layers


def rebuild_network(model, scores, p_filter):
    scores = sorted(scores, key=lambda x: x[0])

    allowed_layers = [x[0] for x in scores]
    scores = [x[1] for x in scores]

    for i in range(0, len(scores)):
        num_remove = round(p_filter * len(scores[i]))
        scores[i] = np.argpartition(scores[i], num_remove)[:num_remove]

    scores = [x for x in zip(allowed_layers, scores)]

    # sort

    if architecture_name.__contains__('ResNet'):
        blocks = rl.count_blocks(model)
        return rebuild_resnet(model=model,
                              blocks=blocks,
                              layer_filters=scores)

    if architecture_name.__contains__('MobileNetV2'):
        blocks = rl.count_blocks(model)
        return rebuild_mobilenetV2(model=model,
                                   blocks=blocks,
                                   layer_filters=scores)

    else:  # If not ResNet nor mobile then it is VGG-Based
        print('TODO: We need to implement (just update) this function')
        return None
