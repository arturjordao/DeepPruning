import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow.keras.utils as keras_utils
import tensorflow.keras.backend as backend
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size':self.patch_size,
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.projection = layers.Dense(units=self.projection_dim)

        # if weights is not None:
        #     self.projection = layers.Dense(units=projection_dim, weights=weights)

        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config

def Transformer(input_shape, projection_dim, num_heads, n_classes, square_patch=8):

    inputs = layers.Input(shape=input_shape)
    patches = Patches(square_patch)(inputs)
    encoded_patches = PatchEncoder((32 // square_patch) ** 2, projection_dim)(patches)

    num_transformer_blocks = len(num_heads)
    for i in range(num_transformer_blocks):

        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads[i], key_dim=projection_dim, dropout=0.0)(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP Size of the transformer layers
        transformer_units = [projection_dim * 2, projection_dim]

        #x3 = FFN(x3, hidden_units=transformer_units)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])


    encoded_patches = layers.Flatten()(encoded_patches)
    if n_classes == 2:
        outputs = layers.Dense(n_classes, activation='sigmoid')(encoded_patches)
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    #return keras.Model(inputs, outputs)
    return Model(inputs, outputs)

def TransformerTabular(input_shape, projection_dim, num_heads, n_classes):

    inputs = layers.Input(shape=input_shape)
    encoded_patches = PatchEncoder(input_shape[0], projection_dim)(inputs)

    num_transformer_blocks = len(num_heads)
    for i in range(num_transformer_blocks):

        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads[i], key_dim=projection_dim, dropout=0.0)(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP Size of the transformer layers
        transformer_units = [projection_dim * 2, projection_dim]

        #x3 = FFN(x3, hidden_units=transformer_units)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])


    encoded_patches = layers.Flatten()(encoded_patches)
    if n_classes == 2:
        outputs = layers.Dense(n_classes, activation='sigmoid')(encoded_patches)
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    #return keras.Model(inputs, outputs)
    return Model(inputs, outputs)