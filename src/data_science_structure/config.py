import torch

# device on which to train
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train size
VAL_SIZE = 0.2

NUM_CLASSES = 2

# embedding sizes
EMB_DIMS = 5

# model layers
OUTPUT_SHAPES = [128]
KERNELS = [23]
STRIDES = [1]


# number of epochs for Training:
N_EPOCHS = 24

# start learning rate
START_LR = 0.001

# start learning rate
END_LR = 0.001

# start learning rate
MAX_LR = 0.01

RUN_PATH = 'run/'