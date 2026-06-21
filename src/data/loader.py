import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df, target_cols, is_train=True):
    text_col = 'comment_text' if 'comment_text' in df.columns else 'text'
    df[text_col] = df[text_col].fillna(" ")

    if is_train and all(col in df.columns for col in target_cols):
        mask = (df[target_cols] != -1).all(axis=1)
        df = df[mask].reset_index(drop=True)

    return df


def load_data(config):
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    sample_sub = pd.read_csv(config.SAMPLE_SUB)

    train_df = clean_data(train_df, config.TARGET_COLS, is_train=True)

    return train_df, test_df, sample_sub


def split_data(df, target_cols, test_size=0.2, random_state=42):
    stratify = df[target_cols].max(axis=1).values
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
