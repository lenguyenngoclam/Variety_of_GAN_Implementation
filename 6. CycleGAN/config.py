import os
# dataset path
TRAIN_X_PATH = 'monet2photo/trainA'
TRAIN_Y_PATH = 'monet2photo/trainB'
VAL_X_PATH = 'monet2photo/testA'
VAL_Y_PATH = 'monet2photo/testB'

IMG_SIZE = 512
IMG_CHANNELS = 3

# Training configuration
BATCH_SIZE = 1
BUFFER_SIZE = 128
SHUFFLE = True
EPOCHS = 200

# Checkpoint path
checkpoint_dir = './checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, 'cycleGan-ckpt')

# Hyperparameter
LEARNING_RATE=1e-5
BETA_1=0.5
BETA_2=0.99

CONSISTENCY_LOSS_LAMBDA = 10