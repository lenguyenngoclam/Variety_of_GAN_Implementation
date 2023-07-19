import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import ZeroPadding2D

class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, padding=(0,0)):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__()
        
    def call(self, x):
        padding_height, padding_width = self.padding
        return tf.pad(x, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], mode='REFLECT')

if __name__ == "__main__":
    reflect_padding = ReflectionPadding2D((1, 2))
    x = tf.random.uniform([1, 3, 3, 3])
    print(x.shape)
    x = reflect_padding(x)
    print(x.shape)