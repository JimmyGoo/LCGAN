Z_SIZE = 100
FIXED_Z = False
WORKERS = 1
MAX_EPOCH = 200
CIFAR_10_PATH = './dataset/' 
LSUN_PATH = '/datag/liushichen/LSUN/'
CIFAR_IMG_SIZE = 32
LSUN_IMG_SIZE = 128
GEN_LEARNING_RATE = 5e-5
DIS_LEARNING_RATE = 5e-5
NUM_SAVE = 64
BATCH_SIZE = 32
LRELU_SLOPE = 0.2
OPTIM = 'adam'
G_LAST = 'tanh'
ALPHA_STEP = 0.05

DECAY = True
SMOOTH_LABEL = True
NORM = False
LAMBDA_NORM = 0.005
NORM_THERSHOLD = 100
CLAMP = False
CLAMP_LOW = -0.01
CLAMP_HIGH = 0.01

EVAL_ITER = 20