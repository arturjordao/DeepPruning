import numpy as np
from sklearn.utils import gen_batches
import keras
from keras.layers.pooling import *
from keras.layers import *
from keras.layers import Input
from keras.models import Model
import gc
import sys
import architecture_ResNetBN as arch

def blocks_to_prune(model):
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
            allowed_layers.append(all_add[i])

    # The last block is enabled
    allowed_layers.append(all_add[-1])
    return allowed_layers

def add_to_downsampling(model):
    layers = []
    all_add = []

    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output_shape
        output_shape = model.get_layer(index=all_add[i - 1]).output_shape
        # These are the downsampling add
        if input_shape != output_shape:
            layers.append(all_add[i])

    return layers

def idx_score_block(blocks, layers):
    #Associates the scores's index with the ResNet block
    output = {}
    idx = 0
    for i in range(0, len(blocks)):
        for layer_idx in range(idx, idx+blocks[i]-1):
            output[layers[layer_idx]] = i
            idx = idx + 1

    return output

def new_blocks(blocks, scores, allowed_layers, p=0.1):
    num_blocks = blocks

    if isinstance(p, float):
        num_remove = round(p * len(scores))
    else:
        num_remove = p

    score_block = idx_score_block(blocks, allowed_layers)
    mask = np.ones(len(allowed_layers))

    #It forces to remove 'num_remove' layers
    i = num_remove
    while i > 0 and not np.all(np.isinf(scores)):
        min_score = np.argmin(scores)
        block_idx = allowed_layers[min_score]#Get the index of the layer associated with the min vip
        block_idx = score_block[block_idx]

        if num_blocks[block_idx]-1 > 1:
            mask[min_score] = 0
            num_blocks[block_idx] = num_blocks[block_idx] - 1

            i = i - 1

        scores[min_score] = np.inf

    return num_blocks, mask

def transfer_weightsBN(model, new_model, mask):
    add_model = blocks_to_prune(model)
    add_new_model = blocks_to_prune(new_model)

    #Add the same weights until finding the first Add layer
    for idx in range(0, len(model.layers)):
        w = model.get_layer(index=idx).get_weights()
        new_model.get_layer(index=idx).set_weights(w)


        if isinstance(model.get_layer(index=idx), Add):
            break

    # These are the layers where the weights must to be transfered
    add_model = np.array(add_model)[mask==1]
    add_model = list(add_model)
    end = len(add_new_model)

    for layer_idx in range(0, end):

        idx_model = np.arange(add_model[0] - 9, add_model[0] + 1).tolist()
        idx_new_model = np.arange(add_new_model[0] - 9, add_new_model[0] + 1).tolist()

        for transfer_idx in range(0, len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)

        add_new_model.pop(0)
        add_model.pop(0)

    # These are the downsampling layers
    add_model = add_to_downsampling(model)
    add_new_model = add_to_downsampling(new_model)

    for i in range(0, len(add_model)):
        idx_model = np.arange(add_model[i] - 11, add_model[i] + 1).tolist()
        idx_new_model = np.arange(add_new_model[i] - 11, add_new_model[i] + 1).tolist()

        for transfer_idx in range(0, len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)

    #This is the dense layer
    w = model.get_layer(index=-1).get_weights()
    new_model.get_layer(index=-1).set_weights(w)

    return new_model

def count_res_blocks(model):
    #Returns the last Add of each block
    res_blocks = {}

    for layer in model.layers:
        if isinstance(layer, keras.layers.Add):
            dim = layer.output_shape[1]#1 and 2 are the spatial dimensions
            res_blocks[dim] = res_blocks.get(dim, 0) + 1

    return list(res_blocks.values())

def rebuild_network(model, scores, p_layer):
    num_classes = model.output_shape[-1]
    input_shape = (model.input_shape[1:])

    allowed_layers = [int(x[0]) for x in scores]
    scores = [x[1] for x in scores]
    blocks = count_res_blocks(model)

    blocks, mask = new_blocks(blocks, scores, allowed_layers, p_layer)

    tmp_model = arch.resnet(input_shape, blocks=blocks, num_classes=num_classes)

    pruned_model = transfer_weightsBN(model, tmp_model, mask)
    return pruned_model