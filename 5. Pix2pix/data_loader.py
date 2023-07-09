import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os

TRAIN_PATH = 'maps/train'
VAL_PATH = 'maps/val'

IMG_SIZE = 256
IMG_CHANNELS = 3

def load_image(image_path):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    img = tf.cast(img, tf.float32)
    
    # Cut image to input and target image
    h, w, _ = img.shape
    w_prime = w // 2
    x, y = img[:, :w_prime, :], img[:, w_prime:, :]
    
    return x, y

def preprocessing_image(x, y):
    # x: input image
    # y: target image
    # I apply the same preprocessing steps as specify in the paper

    # Resize image
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    y = tf.image.resize(y, [IMG_SIZE, IMG_SIZE])
    
    # Random mirroring
    # I create a random number that uniformly distributed in range [0, 1). If this number > 0.5 then i flip both x, y image and otherwise
    random_number = np.random.rand()
    if random_number > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    
    # Scale to range [-1, 1]
    x = (x - 127.5) / 127.5
    y = (y - 127.5) / 127.5
    
    return x, y

def create_dataset(folder_path, batch_size, buffer_size, shuffle=True):
    dataset = tf.data.Dataset.list_files(os.path.join(folder_path, '*.jpg'), seed=1234, shuffle=True)
    print(len(dataset))
    # Configuration for performance
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(lambda x: tf.numpy_function(load_image, [x], [tf.float32, tf.float32]), 
                            num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda inp, tar: tf.numpy_function(preprocessing_image, [inp, tar], [tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
    
    
if __name__ == "__main__":
    
    dataset = create_dataset(TRAIN_PATH, 1, buffer_size=128, shuffle=True)
    for batch in dataset:
        x, y = batch
        print(x.shape, y.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(tf.cast(x[0] * 127.5 + 127.5, tf.uint8))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(tf.cast(y[0] * 127.5 + 127.5, tf.uint8))
        plt.axis('off')
        plt.show()
        break