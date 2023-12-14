# ==========================================
# NLP Project
# ===========================================

import torch
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Hyper-parameters
EPOCHS = 10
SEEDS = [42, 123, 456, 843, 1296]
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LEARNING_RATE = 1e-05

# DataLoader Hyper-paramaters
TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

TEST_PARAMS = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

class EFRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.utterances = dataframe.utterances
        self.triggers = dataframe.triggers
        self.stance = dataframe.speakers
        self.max_len = max_len
        self.len = len(self.data)


def readData(): 
    return pd.read_json('data/MELD_train_efr.json')

def create_dataframes(orginal_df, seed):
    train, test_validation = train_test_split(orginal_df, test_size=0.2, random_state=seed)
    validation, test = train_test_split(test_validation, test_size=0.5, random_state=seed)
    return train, validation, test

def set_default_seed():
    _seed = SEEDS[0]

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    random.seed(_seed)
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)

    return _seed

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device => ", device)

def prepare_data():
    pass


def main():
    _df = readData()
    
    seed = set_default_seed()
    set_device()

    df_train, df_validation, df_test = create_dataframes(_df, seed)


    print(_df['speakers'][0])
    print(_df['utterances'][0])
    print(_df['triggers'][0])
    print(_df.keys())
    print(len(_df))
    prepare_data()


if __name__ == "__main__":
    main()