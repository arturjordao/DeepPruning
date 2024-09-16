import argparse
import os
import random
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score
from tensorflow.python.data import Dataset

import rebuild_heads as rh
import rebuild_layers as rl
import template_architectures
from pruning_criteria import criteria_head as ch
from pruning_criteria import criteria_layer as cl

sys.path.insert(0, '../utils')
import custom_functions as func
import custom_callbacks


# import utils.custom_functions as func
# import utils.custom_callbacks as custom_callbacks

def load_transformer_data(file='synthetic'):
    """
    loads premade data for transformer model
    Args:
        file: name of the file to load data from; 'synthetic' generates random data for tests
    Returns:
        x_train, x_test, y_train, y_test, n_classes
    """
    if file == 'synthetic':
        # Synthetic data"
        samples, features, n_classes = 1000, 200, 3
        x_train, x_test = np.random.rand(samples, features), np.random.rand(int(samples / 10),
                                                                            features)  # samples x features
        y_train = np.random.randint(0, n_classes, len(x_train))
        y_test = np.random.randint(0, n_classes, len(x_test))
        n_classes = len(np.unique(y_train, axis=0))
    else:
        # Real data -- DecaLearn
        if '.npz' not in file:
            file += '.npz'

        tmp = np.load(file)
        x_train, x_test, y_train, y_test = tmp['X_train'], tmp['X_test'], tmp['y_train'], tmp['y_test']

        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)

        n_classes = len(np.unique(y_train, axis=0))
        y_train = np.eye(n_classes)[y_train]
        y_test = np.eye(n_classes)[y_test]

    return x_train, x_test, y_train, y_test, n_classes


def flops(model, verbose=False):
    """
    Calculate FLOPS used by the model
    Args:
        model: model to calculate perfomance of
        verbose: if True, returns extra information (I.E.: flops per type of operation among others)

    Returns:
        numbers of flops used by the model
    """
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        if not verbose:
            opts['output'] = 'none'
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


def statistics(model):
    """
    prints statistics of the model: number of heads, parameters, FLOPS and memory
    Args:
        model: model from which the statistics are calculated
    """
    n_params = model.count_params()
    n_heads = func.count_filters(model)
    memory = func.memory_usage(1, model)
    tmp = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]

    print('#Heads {} Params [{}]  FLOPS [{}] Memory [{:.6f}]'.format(tmp, n_params, flops(model), memory), flush=True)


def fine_tuning(model, x_train, y_train, x_test, y_test):
    if len(model.input.shape.as_list()) == 3:
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=512, verbose=0, epochs=10)

    else:
        batch_size = 1024
        lr = 0.001
        schedule = [(100, lr / 10), (150, lr / 100)]
        lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
        callbacks = [lr_scheduler]

        sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        for ep in range(0, 200):
            y_tmp = np.concatenate((y_train, y_train, y_train))
            x_tmp = np.concatenate(
                (func.data_augmentation(x_train),
                 func.data_augmentation(x_train),
                 func.data_augmentation(x_train)))

            x_tmp = Dataset.from_tensor_slices((x_tmp, y_tmp)).shuffle(4 * batch_size).batch(batch_size)

            model.fit(x_tmp, batch_size=batch_size, verbose=2,
                      callbacks=callbacks,
                      epochs=ep, initial_epoch=ep - 1)

            if ep % 5 == 0:  # % 5
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test, verbose=0), axis=1))
                print('Accuracy [{:.4f}]'.format(acc), flush=True)
                # func.save_model('TransformerViT_epoch[{}]'.format(ep), model)

    return model


if __name__ == '__main__':
    np.random.seed(12227)
    random.seed(12227)

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
    os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
    physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_name', type=str, default='')
    parser.add_argument('--criterion_head', type=str, default='random')
    parser.add_argument('--criterion_layer', type=str, default='random')
    parser.add_argument('--p_head', type=float, default=1)
    parser.add_argument('--p_layer', type=int, default=1)

    args = parser.parse_args()
    architecture_name = args.architecture_name
    criterion_head = args.criterion_head
    criterion_layer = args.criterion_layer
    p_head = args.p_head
    p_layer = args.p_layer
    tabular = False

    print(args, flush=False)
    print('Architecture [{}] p_head[{}] p_layer[{}]'.format(architecture_name, p_head, p_layer), flush=True)

    # Image######################
    if not tabular:
        x_train, y_train, x_test, y_test = func.cifar_resnet_data(debug=True)

        input_shape = x_train.shape[1:]
        n_classes = y_train.shape[1]

        # model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        # model.fit(X_train, y_train, batch_size=256, verbose=2, epochs=10)
        model = func.load_transformer_model('TransformerViT', 'TransformerViT')

    # Tabular################
    else:
        x_train, x_test, y_train, y_test, n_classes = load_transformer_data(file='FaciesClassificationYananGasField')

        input_shape = (x_train.shape[1:])
        projection_dim = 64
        num_heads = [256, 128, 64, 16, 32, 8]

        model = template_architectures.TransformerTabular(input_shape, projection_dim, num_heads, n_classes)
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        model = fine_tuning(model, x_train, y_train, x_test, y_test)

    y_pred = model.predict(x_test)
    acc = accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
    print('Accuracy Unpruned [{}]'.format(acc))
    statistics(model)

    for i in range(0, 40):
        prob = random.random()  # Flipping a coin
        if prob >= 0.5:
            head_method = ch.criteria(criterion_head)
            scores = head_method.scores(model, x_train, y_train, rh.heads_to_prune(model))
            model = rh.rebuild_network(model, scores, p_head)
        else:
            layer_method = cl.criteria(criterion_layer)
            scores = layer_method.scores(model, x_train, y_train, rl.layers_to_prune(model))
            model = rl.rebuild_network(model, scores, p_layer)

        # Uncomment to real experiments
        model = fine_tuning(model, x_train, y_train, x_test, y_test)
        acc = accuracy_score(np.argmax(model.predict(x_test, verbose=0), axis=1), np.argmax(y_test, axis=1))
        statistics(model)
        print('Acc [{}]'.format(acc))
        print('Iteration [{}] Accuracy [{}]'.format(i, acc))
        # func.save_model('{}_{}_p[{}]_iterations[{}]'.format(architecture_name, criterion_layer, p_layer, i), model)
