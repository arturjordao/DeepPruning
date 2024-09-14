import copy

import numpy as np
from keras import layers

import template_architectures


def heads_to_prune(model):
    output = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.MultiHeadAttention):
            output.append(i)

    return output


def rw_multi_head(w, idx):
    w_new = copy.deepcopy(w)
    for i in range(0, 6):
        w_new[i] = np.delete(w_new[i], idx, axis=1) if i % 2 == 0 else np.delete(w_new[i], idx, axis=0)

    w_new[-2] = np.delete(w_new[-2], idx, axis=0)
    return w_new


def unimportant_heads(scores, p):
    output = []
    layers = [x[0] for x in scores]
    scores = [x[1] for x in scores]

    for i in range(len(layers)):
        if isinstance(p, float):
            num_remove = round(p * len(scores[i]))
        else:
            num_remove = p

        num_heads = len(scores[i])
        # It forces to remove 'num_remove' heads

        if num_heads == 1:
            tmp = []
        else:
            num_remove = min(num_remove, num_heads - 1)
            tmp = np.argpartition(scores[i], num_remove)[:num_remove]

        output.append((layers[i], tmp))

    return output


def rebuild_network(model, scores, p):
    input_shape = model.input_shape[1:]
    projection_dim = [layer._key_dim for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)][0]
    n_classes = model.get_layer(index=-1).output_shape[-1]

    # According to the scores, this function selects the head to be eliminated
    layer_heads = unimportant_heads(copy.deepcopy(scores), p)

    # Avoids empty multiheads (its says 0 heads) - Begin
    num_heads = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]
    layer_heads.sort(key=lambda tup: tup[0])  # Ensures that pop(ith) corresponds to ith (attention) layer
    num_heads_new = [x[0] - len(x[1][1]) for x in zip(num_heads, layer_heads)]

    for i in range(0, len(layer_heads)):
        if num_heads_new[i] == 0:
            layer_heads[i] = (layer_heads[i][0], [])
            num_heads_new[i] = num_heads[i]
    # End

    if len(model.input.shape.as_list()) == 3:  # Tabular
        pruned_model = template_architectures.TransformerTabular(input_shape, projection_dim, num_heads_new, n_classes)
    else:
        pruned_model = template_architectures.Transformer(input_shape, projection_dim, num_heads_new, n_classes)

    remove_head = [item[1] for item in layer_heads]

    # After we build the model, we assign the weighs to multihead layers
    # Currently, tf+keras does not enable we assing weights during layer creation
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.MultiHeadAttention):
            w = layer.get_weights()
            w = rw_multi_head(w, remove_head.pop(0))
            pruned_model.get_layer(index=i).set_weights(w)
        else:
            w = layer.get_weights()
            pruned_model.get_layer(index=i).set_weights(w)

    return pruned_model
