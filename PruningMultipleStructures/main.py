import numpy as np
import copy
from sklearn.metrics._classification import accuracy_score
import sys
from keras.layers import *
import keras.backend as K
from keras.activations import *
import argparse
import gc
import rebuild_layers as rl
import rebuild_filters as rf
from pruning_criteria import criteria_filter as cf
from pruning_criteria import criteria_layer as cl
import template_architectures

sys.path.insert(0, '../utils')
import custom_functions as func
import custom_callbacks

def statistics(model):
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_blocks(model)

    memory = func.memory_usage(1, model)
    print('Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
          'Memory [{:.6f}]'.format(blocks, n_params, n_filters, flops, memory), flush=True)

if __name__ == '__main__':
    np.random.seed(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_name', type=str, default='ResNet56')
    parser.add_argument('--criterion_filter', type=str,default='random')
    parser.add_argument('--criterion_layer', type=str, default='random')
    parser.add_argument('--p_filter', type=float, default=0.1)
    parser.add_argument('--p_layer', type=int, default=1)

    args = parser.parse_args()
    architecture_name = args.architecture_name
    criterion_filter = args.criterion_filter
    criterion_layer = args.criterion_layer
    p_filter = args.p_filter
    p_layer = args.p_layer

    print(args, flush=False)
    print('Architecture [{}] p_filter[{}] p_layer[{}]'.format(architecture_name, p_filter, p_layer), flush=True)

    X_train, y_train, X_test, y_test, X_val, y_val = func.cifar_resnet_data(debug=True, validation_set=True)

    model = func.load_model('{}'.format(architecture_name), '{}'.format(architecture_name))

    rf.architecture_name = architecture_name
    rl.architecture_name = architecture_name

    statistics(model)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
    print('Unpruned [{}] Accuracy [{}]'.format(architecture_name, acc))

    for i in range(0, 15):

        # #Filter
        allowed_layers_filters = rf.layer_to_prune_filters(model)
        filter_method = cf.criteria(criterion_filter)
        scores = filter_method.scores(model, X_train, y_train, allowed_layers_filters)
        pruned_model_filter = rf.rebuild_network(model, scores, p_filter)

        allowed_layers = rl.blocks_to_prune(model)
        layer_method = cl.criteria(criterion_layer)
        scores = layer_method.scores(model, X_train, y_train, allowed_layers)
        pruned_model_layer = rl.rebuild_network(model, scores, p_layer)


        prob = np.random.rand(1)[0]
        if prob >= 0.5:
            model = pruned_model_layer
            structure = 'Layer'
        else:
            model = pruned_model_filter
            structure = 'Filter'

        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test,verbose=0), axis=1))
        statistics(model)
        print('Iteration [{}] Structure [{}] Accuracy [{}]'.format(i, structure, acc))