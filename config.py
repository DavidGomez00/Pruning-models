# Training hiperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 3
MIN_EPOCHS = 1
#K-Fold
NUM_FOLDS = 5
SPLIT_SEED = 42

# Dataset
DATA_DIR = "perrosygatos/"
NUM_WORKERS = 7

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "bf16-mixed"
