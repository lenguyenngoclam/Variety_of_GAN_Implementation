import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose, Activation, \
        ReLU, LeakyReLU, BatchNormalization, Identity
from tensorflow.keras.activations import tanh
from custom_padding_layers import ReflectionPadding2D
from tensorflow.keras.initializers import RandomNormal
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
import config

class Block(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, down:bool=True, use_act:bool=True):
        super(Block, self).__init__()
        self.net = keras.models.Sequential(
            [
                # The code implemented here is clumsy because keras only support zero padding so i have to create custom reflection padding for down sampling block.
                ReflectionPadding2D(padding=padding) if down else Identity(),
                
                Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02)) 
                if down else
                Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same',
                                kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                
                BatchNormalization(axis=[0,-1]),
                ReLU() if use_act else Identity()
            ]
        )
    def call(self, x, training=True):
        return self.net(x)
    
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.block = keras.models.Sequential(
            [
                Block(filters=filters, kernel_size=3, strides=1, padding=(1, 1)),
                Block(filters=filters, kernel_size=3, strides=1, padding=(1, 1), use_act=False)
            ]
        )
    def call(self, x, training=True):
        return x + self.block(x)
    
class Generator(keras.Model):
    def __init__(self, down_features=[128, 256], num_res=9, up_features=[128, 64]):
        super(Generator, self).__init__()
        self.initial = keras.models.Sequential(
            [
                ReflectionPadding2D(padding=(3, 3)),
                Conv2D(filters=64, kernel_size=7, strides=1,
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                ReLU()
            ]
        )
        self.downs = keras.models.Sequential([Block(filters=filters,
                                                    kernel_size=3,
                                                    strides=2,
                                                    padding=(1, 1)) for filters in down_features])
        self.res_blocks = keras.models.Sequential([ResidualBlock(filters=256) for _ in range(num_res)])
        self.ups = keras.models.Sequential([Block(filters=filters, 
                                                  kernel_size=3, 
                                                  strides=2, 
                                                  padding=(1, 1),
                                                  down=False) for filters in up_features])
        
        self.output_layer = keras.models.Sequential(
            [
                Conv2DTranspose(filters=config.IMG_CHANNELS, kernel_size=7, strides=1, padding='same',
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                          bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                Activation(tanh)
            ]
        )
        
    def call(self, x):
        x = self.initial(x)
        x = self.downs(x)
        x = self.res_blocks(x)
        x = self.ups(x)
        return self.output_layer(x)

if __name__ == "__main__":
    # Check the validity of generator
    gen = Generator()
    x = tf.random.uniform([1, 256, 256, config.IMG_CHANNELS])
    print(x.shape)
    x = gen.predict(x)
    print(x.shape)