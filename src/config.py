import os

# Lokasi Folder Data
RAW_DATA_DIR = 'data/raw/garbage_classification'
PROCESSED_DATA_DIR = 'data/processed'

# Rasio Splitting (70% Train, 15% Val, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Parameter Model & Dataset
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 123

# Training fase 1
EPOCHS_INITIAL = 15
LR_INITIAL = 1e-3

# Training fase 2 (fine-tuning)
EPOCHS_FINETUNE = 10
LR_FINETUNE = 1e-5

# Early stopping
PATIENCE_INITIAL = 5
PATIENCE_FINETUNE = 3