import sys

import tensorflow as tf
from keras import layers
from keras.models import Model

sys.path.insert(0, '../utils')
from custom_classes import Patches, PatchEncoder


def Transformer(input_shape, projection_dim, num_heads, n_classes, square_patch=8):
    inputs = layers.Input(shape=input_shape)
    patches = Patches(square_patch)(inputs)
    encoded_patches = PatchEncoder((32 // square_patch) ** 2, projection_dim)(patches)

    num_transformer_blocks = len(num_heads)
    for i in range(num_transformer_blocks):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads[i], key_dim=projection_dim, dropout=0.0)(x1,
                                                                                                                  x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP Size of the transformer layers
        # transformer_units = [projection_dim * 2, projection_dim]

        # x3 = FFN(x3, hidden_units=transformer_units)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    encoded_patches = layers.Flatten()(encoded_patches)
    if n_classes == 2:
        outputs = layers.Dense(n_classes, activation='sigmoid')(encoded_patches)
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    # return keras.Model(inputs, outputs)
    return Model(inputs, outputs)


def TransformerTabular(input_shape, projection_dim, num_heads, n_classes):
    inputs = layers.Input(shape=input_shape)
    encoded_patches = PatchEncoder(input_shape[0], projection_dim)(inputs)

    num_transformer_blocks = len(num_heads)
    for i in range(num_transformer_blocks):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads[i], key_dim=projection_dim, dropout=0.0)(x1,
                                                                                                                  x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP Size of the transformer layers
        # transformer_units = [projection_dim * 2, projection_dim]

        # x3 = FFN(x3, hidden_units=transformer_units)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    encoded_patches = layers.Flatten()(encoded_patches)
    if n_classes == 2:
        outputs = layers.Dense(n_classes, activation='sigmoid')(encoded_patches)
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    return Model(inputs, outputs)
