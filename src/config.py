import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    DATA_DIR = os.environ.get("DATA_DIR", "data")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

    TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
    TEST_FILE = os.path.join(DATA_DIR, "test.csv")
    SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

    TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    MAX_LENGTH = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5


class RobertaConfig(Config):
    MODEL_NAME = "distilroberta-base"
    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 128
    DROPOUT = 0.2


class TextCNNConfig(Config):
    EMBEDDING_DIM = 100
    MAX_VOCAB_SIZE = 20000
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    HIDDEN_DIM = 128
    DROPOUT = 0.5
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
