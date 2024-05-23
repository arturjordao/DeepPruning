import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.utils import gen_batches
import sys
import keras
import tensorflow as tf
from tensorflow.data import Dataset
import h5py
from tensorflow.keras.applications.resnet50 import preprocess_input
import argparse

import rebuild_layers as rl
from pruning_criteria import criteria_layer as cl

sys.path.insert(0, '../utils')

import custom_functions as func
import custom_callbacks
import architecture_ResNetBN as arch


def statistics(model):
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    filter_layer = func.count_filters_layer(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_res_blocks(model)

    memory = func.memory_usage(1, model)
    print('Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
          'Memory [{:.6f}]'.format(blocks, n_params, n_filters, flops, memory), flush=True)


def prediction(model, X_test, y_test):
    y_pred = np.zeros((X_test.shape[0], y_test.shape[1]))

    for batch in gen_batches(X_test.shape[0], 256):  # 256 stands for the number of samples in primary memory
        samples = preprocess_input(X_test[batch].astype(float))

        # with tf.device("CPU"):
        X_tmp = Dataset.from_tensor_slices((samples)).batch(256)

        y_pred[batch] = model.predict(X_tmp, batch_size=256, verbose=0)

    top1 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=1)
    top5 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=5)
    top10 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=10)
    print('Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.format(top1, top5, top10), flush=True)


def finetuning(model, X_train, y_train, X_test, y_test):
    lr = 0.001
    schedule = [(2, 1e-4), (4, 1e-5)]

    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    for ep in range(0, 5):

        for batch in gen_batches(X_train.shape[0], 1024):
            samples = func.data_augmentation(X_train[batch].astype(float), padding=28)
            samples = preprocess_input(samples)

            #with tf.device("CPU"):
            X_tmp = Dataset.from_tensor_slices((samples, y_train[batch])).shuffle(4 * 64).batch(64)

            model.fit(X_tmp,
                      callbacks=callbacks, verbose=2,
                      epochs=ep, initial_epoch=ep - 1,
                      batch_size=64)
        if ep % 3:
            prediction(model, X_test, y_test)


        func.save_model('Criterion[{}]' + '_Blocks{}_P[{}]_Epoch{}'.format(criterion_layer, blocks, p_layer, ep), model)

    return model


if __name__ == '__main__':
    np.random.seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet50')
    parser.add_argument('--criterion_layer', type=str, default='random')
    parser.add_argument('--p_layer', type=float, default=1)
    debug = True

    args = parser.parse_args()
    architecture_name = args.architecture
    p_layer = args.p_layer
    criterion_layer = args.criterion_layer
    # scores = []#Put the precomputed scores here
    cl.preprocess_input = preprocess_input

    print('Architecture [{}] Criterion[{}] P[{}]'.format(architecture_name, criterion_layer, p_layer), flush=True)

    if debug == False:
        model = func.load_model(architecture_name, architecture_name)
        tmp = h5py.File('E:/ImageNet/imageNet_images.h5', 'r')
        X_train, y_train = tmp['X_train'], tmp['y_train']
        X_test, y_test = tmp['X_test'], tmp['y_test']
        cl.n_samples = 10

    else:
        n_samples = 100
        resolution = 224
        X_train = np.random.rand(n_samples, resolution, resolution, 3)#TODO:Ensure at least one sample per class
        X_test = np.random.rand(n_samples, resolution, resolution, 3)

        y_train = np.eye(1000)[np.random.randint(0, 1000, n_samples)]
        y_test = np.eye(1000)[np.random.randint(0, 1000, n_samples)]

    while rl.count_res_blocks(model) != [2, 2, 2, 2]:
        allowed_layers = rl.blocks_to_prune(model)
        layer_method = cl.criteria(criterion_layer)
        scores = layer_method.scores(model, X_train, y_train, allowed_layers)

        model = rl.rebuild_network(model, scores, p_layer)

        # model = finetuning(model, X_train, y_train, X_test, y_test)
        statistics(model)
        prediction(model, X_test, y_test)
        # func.save_model('Criterion[{}]' + '_Blocks{}_P[{}]'.format(criterion_layer, blocks, p_layer), model)