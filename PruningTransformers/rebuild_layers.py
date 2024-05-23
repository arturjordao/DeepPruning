import numpy as np
from tensorflow.keras import layers

import template_architectures

def count_blocks(model):
    output = 0

    for layer in model.layers:
        if isinstance(layer, layers.MultiHeadAttention):
            output = output + 1

    return output

def layers_to_prune(model):
    output = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.MultiHeadAttention):
            output.append(i-1)#We index the first layer normalization

    output.pop(0)#We can't prune the first block
    return output

def new_blocks(blocks, scores, allowed_layers, p):
    num_blocks = blocks

    if isinstance(p, float):
        num_remove = round(p * len(scores))
    else:
        num_remove = p

    mask = np.ones(len(allowed_layers))

    #It forces to remove 'num_remove' layers
    i = num_remove
    while i > 0 and not np.all(np.isinf(scores)):
        min_ = np.argmin(scores)#Finds the minimum score
        if num_blocks-1 > 1:
            mask[min_] = 0
            num_blocks = num_blocks - 1
            i = i - 1

        scores[min_] = np.inf

    return num_blocks, mask

def heads_layer(model, mask, allowed_layers):
    output = []
    add_number = 0

    for i,layer in enumerate(model.layers):

        if isinstance(layer, layers.MultiHeadAttention):
            output.append((i, layer._num_heads))

        if isinstance(layer, layers.Add):
            add_number = add_number + 1
        if add_number == 2:#The second add means that we need to stop
            break

    tmp = allowed_layers#layers_to_prune(model)
    tmp = np.array(tmp)[mask==1]

    tmp = tmp + 1
    for layer_idx in tmp:
        output.append((layer_idx, model.get_layer(index=layer_idx)._num_heads))

    output = list(set(output)) #Remove duplicates
    output.sort(key=lambda tup: tup[0])
    output = [item[1] for item in output]
    return output

def transfer_weights(model, new_model, mask, allowed_layers):

    assigned_weights = np.zeros((len(new_model.layers)), dtype=bool)
    add_number = 0

    for i, layer in enumerate(model.layers):
        w = model.get_layer(index=i).get_weights()
        new_model.get_layer(index=i).set_weights(w)
        assigned_weights[i] = True

        if isinstance(layer, layers.Add):
            add_number = add_number + 1
        if add_number == 2:#The second add means that we need to stop
            break

    layers_model = list(np.array(allowed_layers)[mask==1])
    layers_new_model = layers_to_prune(new_model)

    for _ in range(0, len(layers_new_model)):
        idx_model = np.arange(layers_model[0], layers_model[0]+7).tolist()
        idx_new_model = np.arange(layers_new_model[0], layers_new_model[0]+7).tolist()

        for transfer_idx in range(0, len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)

        assigned_weights[idx_new_model] = True
        layers_model.pop(0)
        layers_new_model.pop(0)

    i = len(model.layers) - 1
    j = len(new_model.layers) - 1
    #We assign the weights until finding an Add layers -- from back to front
    while not isinstance(model.get_layer(index=i), layers.Add):
        w = model.get_layer(index=i).get_weights()
        new_model.get_layer(index=j).set_weights(w)
        assigned_weights[j] = True
        i = i - 1
        j = j - 1

    for i in range(0, len(assigned_weights)):
        if assigned_weights[i] == False:
            print('Caution! Weights from Layer[{}] were not transferred'.format(i))

    return new_model

def rebuild_network(model, scores, p):
    blocks = count_blocks(model)

    allowed_layers = [x[0] for x in scores]
    scores = [x[1] for x in scores]

    blocks_tmp, mask = new_blocks(blocks, np.copy(scores), allowed_layers, p)

    heads = heads_layer(model, mask, allowed_layers)
    n_classes = model.get_layer(index=-1).output_shape[-1]

    projection_dim = [layer._key_dim for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)][0]

    input_shape = model.input_shape[1:]
    if len(model.input.shape.as_list()) == 3: #Tabular
        tmp_model = template_architectures.TransformerTabular(input_shape, projection_dim, heads, n_classes)
    else:
        tmp_model = template_architectures.Transformer(input_shape, projection_dim, heads, n_classes)

    model = transfer_weights(model, tmp_model, mask, allowed_layers)
    return model