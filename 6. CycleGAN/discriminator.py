import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from custom_padding_layers import ReflectionPadding2D
from typing import List
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

class Block(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(Block, self).__init__()
        self.block = keras.models.Sequential(
            [
                ReflectionPadding2D(padding=padding),
                Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                BatchNormalization(axis=[0,-1]),
                LeakyReLU(alpha=0.2)
            ]
        )
    def call(self, x):
        return self.block(x)
    
class Discriminator(keras.Model):
    def __init__(self, features:List[int]=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = keras.models.Sequential(
            [
                ReflectionPadding2D(padding=(1, 1)),
                Conv2D(filters=features[0], kernel_size=4, strides=2,
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                LeakyReLU(alpha=0.2)
            ]
        )
        
        layers = []
        for feature in features[1:]:
            layers.append(Block(filters=feature, kernel_size=4,
                                strides=1 if feature == features[-1] else 2, 
                                padding=(1, 1)))
            
        self.conv_blocks = keras.models.Sequential(layers)
        
        self.output_layer = keras.models.Sequential(
            [
                ReflectionPadding2D(padding=(1, 1)),
                Conv2D(filters=1, kernel_size=4, strides=2, activation='sigmoid',
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0.0, stddev=0.02))
            ]
        )
        
    def call(self, x):
        x = self.initial(x)
        x = self.conv_blocks(x)
        return self.output_layer(x)
    
if __name__ == "__main__":
    # Check for validity of discriminator
    inp = tf.random.normal([1, 256, 256, 3])
    print(inp.shape)
    dis = Discriminator()
    out = dis.predict(inp)
    print(out.shape)