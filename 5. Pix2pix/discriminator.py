import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Activation, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import RandomNormal
from typing import List

IMG_CHANNELS = 3

class Block(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding):
        super(Block, self).__init__()
        
        self.conv_block = keras.models.Sequential(
            [
                Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                BatchNormalization(),
                LeakyReLU(alpha=0.2)
            ]
        )
    
    def call(self, x):
        return self.conv_block(x)
    
class Discriminator(keras.Model):
    def __init__(self, features:List[int]=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        
        # We will not apply BatchNormalization to the first layer
        self.initial = keras.models.Sequential(
            [
                Conv2D(filters=features[0], kernel_size=4, strides=2, padding='same',
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                LeakyReLU(alpha=0.2)
            ]
        )
        
        layers = []
        for feature in features[1:]:
            layers.append(Block(filters=feature,
                                kernel_size=4,
                                strides=1 if feature == features[-1] else 2, 
                                padding='valid' if feature == features[-1] else 'same'))
            
        layers.append(Conv2D(filters=1, kernel_size=4, strides=2, padding='valid', activation='sigmoid',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                            bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        self.net = keras.models.Sequential(layers)
        
    def call(self, x, y):
        x = tf.concat([x, y], axis=-1)
        x = self.initial(x)
        
        # The receptive field will be 70x70
        return self.net(x)
    
if __name__ == "__main__":
    inp = tf.random.uniform([1, 256, 256, IMG_CHANNELS])
    
    disc = Discriminator()
    output = disc(inp, inp)
    print(output.shape)