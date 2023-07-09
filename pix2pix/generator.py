import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Dropout, Activation
from tensorflow.keras.activations import tanh
from tensorflow.keras.initializers import RandomNormal
    
IMG_CHANNELS = 3    

class DownSamplingBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(DownSamplingBlock, self).__init__()
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
    
class UpSamplingBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, dropout=False):
        super(UpSamplingBlock, self).__init__()
        self.conv_block = keras.models.Sequential(
            [
                Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                BatchNormalization()
            ]
        )
        if dropout:
            self.conv_block.add(Dropout(0.5))
        self.conv_block.add(ReLU())
        
    def call(self, x):
        return self.conv_block(x)

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # Define all of encoder blocks
        # First layer in encoder will not be applied BatchNormalization 
        self.down1 = keras.models.Sequential(
            [
                Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                LeakyReLU(alpha=0.2)
            ]
        )
        self.down2 = DownSamplingBlock(filters=128, kernel_size=4, strides=2, padding='same')
        self.down3 = DownSamplingBlock(filters=256, kernel_size=4, strides=2, padding='same')
        self.down4 = DownSamplingBlock(filters=512, kernel_size=4, strides=2, padding='same')
        self.down5 = DownSamplingBlock(filters=512, kernel_size=4, strides=2, padding='same')
        self.down6 = DownSamplingBlock(filters=512, kernel_size=4, strides=2, padding='same')
        self.down7 = DownSamplingBlock(filters=512, kernel_size=4, strides=2, padding='same')
        self.down8 = keras.models.Sequential(
            [
                Conv2D(filters=512, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                      bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                ReLU()
            ]
        )
        
        # Define all of decoder blocks
        self.up1 = UpSamplingBlock(filters=512, kernel_size=4, strides=2, padding='same', dropout=True)
        self.up2 = UpSamplingBlock(filters=1024, kernel_size=4, strides=2, padding='same', dropout=True)
        self.up3 = UpSamplingBlock(filters=1024, kernel_size=4, strides=2, padding='same', dropout=True)
        self.up4 = UpSamplingBlock(filters=1024, kernel_size=4, strides=2, padding='same')
        self.up5 = UpSamplingBlock(filters=512, kernel_size=4, strides=2, padding='same')
        self.up6 = UpSamplingBlock(filters=256, kernel_size=4, strides=2, padding='same')
        self.up7 = UpSamplingBlock(filters=128, kernel_size=4, strides=2, padding='same')
        
        # Define output layer
        self.output_layer = keras.models.Sequential(
            [
                Conv2DTranspose(filters=IMG_CHANNELS, kernel_size=4, strides=2, padding='same',
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                Activation(tanh)
            ]
        )
        
    def call(self, x):
        # Encoder
        d1 = self.down1(x) # 128x128
        d2 = self.down2(d1) # 64x64
        d3 = self.down3(d2) # 32x32
        d4 = self.down4(d3) # 16x16
        d5 = self.down5(d4) # 8x8
        d6 = self.down6(d5) # 4x4
        d7 = self.down7(d6) # 2x2
        bottle_kneck = self.down8(d7) # 1x1
        
        # Decoder
        u1 = self.up1(bottle_kneck)
        u2 = self.up2(tf.concat([u1, d7], axis=-1))
        u3 = self.up3(tf.concat([u2, d6], axis=-1))
        u4 = self.up4(tf.concat([u3, d5], axis=-1))
        u5 = self.up5(tf.concat([u4, d4], axis=-1))
        u6 = self.up6(tf.concat([u5, d3], axis=-1))
        u7 = self.up7(tf.concat([u6, d2], axis=-1))
        
        return self.output_layer(tf.concat([u7, d1], axis=-1))
    
if __name__ == "__main__":
    # Check for validity
    x = tf.random.uniform([1, 256, 256, 3])
    gen = Generator()
    
    output = gen(x)
    print(output.shape)