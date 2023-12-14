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

def readData(): 
    return pd.read_json('data/MELD_train_efr.json')

def create_dataframes(orginal_df, seed):
    train, test_validation = train_test_split(orginal_df, test_size=0.2, random_state=seed)
    validation, test = train_test_split(test_validation, test_size=0.5, random_state=seed)
    return train, validation, test

def set_default_seed():
    SEEDS = [42, 123, 456]
    _seed = SEEDS[0]
    
    # Seed Configs
    random.seed(_seed)
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    os.environ['PYTHONHASHSEED']=str(_seed)

    return _seed


def prepare_data():
    pass


def main():
    _df = readData()
    
    seed = set_default_seed()

    df_train, df_validation, df_test = create_dataframes(_df, seed)

    print(_df['speakers'][0])
    print(_df['utterances'][0])
    print(_df['triggers'][0])
    print(_df.keys())
    print(len(_df))
    prepare_data()


if __name__ == "__main__":
    main()