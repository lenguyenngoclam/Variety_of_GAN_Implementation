import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
import config
import cv2

class DataLoader:
    def __init__(self, X_domain_folder_path, Y_domain_folder_path):
        self.X_domain_folder_path = X_domain_folder_path
        self.Y_domain_folder_path = Y_domain_folder_path
        
        self.X_image_paths = os.listdir(self.X_domain_folder_path)
        self.Y_image_paths = os.listdir(self.Y_domain_folder_path)
        
        self.X_len = len(self.X_image_paths)
        self.Y_len = len(self.Y_image_paths)
        self.max_len = max(len(self.X_image_paths), len(self.Y_image_paths))
    
    def load_image(self, x_path, y_path):
        # Read X image
        x_img = tf.io.read_file(os.path.join(self.X_domain_folder_path, x_path.decode("utf-8")))
        x_img = tf.image.decode_jpeg(x_img, channels=config.IMG_CHANNELS)
        x_img = tf.cast(x_img, tf.float32)
        

        # Read Y image
        y_img = tf.io.read_file(os.path.join(self.Y_domain_folder_path, y_path.decode("utf-8")))
        y_img = tf.image.decode_jpeg(y_img, channels=config.IMG_CHANNELS)
        y_img = tf.cast(y_img, tf.float32)
        
        return x_img, y_img
    
    def preprocessing_image(self, x_img, y_img):
        # I apply the same preprocessing steps as specify in the paper

        # Resize image
        x_img = tf.image.resize(x_img, [config.IMG_SIZE, config.IMG_SIZE])
        y_img = tf.image.resize(y_img, [config.IMG_SIZE, config.IMG_SIZE])
        
        # Random mirroring
        # I create a random number that uniformly distributed in range [0, 1). If this number > 0.5 then i flip both x, y image and otherwise
        random_number = np.random.rand()
        if random_number > 0.5:
            x_img = tf.image.flip_left_right(x_img)
            y_img = tf.image.flip_left_right(y_img)

        # Scale to range [-1, 1]
        x_img = (x_img - 127.5) / 127.5
        y_img = (y_img - 127.5) / 127.5

        return x_img, y_img
    
    def get_paths(self, idx):
        return self.X_image_paths[idx % self.X_len], self.Y_image_paths[idx % self.Y_len]

    def get_dataset(self, batch_size, buffer_size, shuffle=True):
        dataset = tf.data.Dataset.range(self.max_len)
        
        dataset = dataset.map(lambda idx: tf.numpy_function(self.get_paths, [idx], [tf.string, tf.string]))
        dataset = dataset.map(lambda x_path, y_path: tf.numpy_function(self.load_image, [x_path, y_path],
                                                                      [tf.float32, tf.float32]),
                                                             num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x_img, y_img: tf.numpy_function(self.preprocessing_image,
                                                                     [x_img, y_img],
                                                                     [tf.float32, tf.float32]),
                                                                    num_parallel_calls=tf.data.AUTOTUNE)
                                                                     
        # Configuration for performance
        # Uncomment if your RAM is large
        # dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
                              
        return dataset
    
if __name__ == "__main__":
    loader = DataLoader(config.TRAIN_X_PATH, config.TRAIN_Y_PATH)
    dataset = loader.get_dataset(config.BATCH_SIZE, config.BUFFER_SIZE, config.SHUFFLE)
    
    for batch in dataset:
        x, y = batch
        print(x.shape, y.shape)
        
        img = cv2.hconcat([tf.cast(x[0] * 127.5 + 127.5, tf.uint8).numpy(),
                           tf.cast(y[0] * 127.5 + 127.5, tf.uint8).numpy()])
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        break
    print('Done!')