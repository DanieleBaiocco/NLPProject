# ==========================================
# NLP Project
# ===========================================
# finding the problem of dataset
# add group idx to df
# add speaker idx to df
# create bert class 
# try to find out best way for managing classes

import torch
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from functools import cache
import time
import argparse
from pathlib import Path
import logging
import copy
import pickle
from train_val import EFRTraining
from sklearn.preprocessing import MultiLabelBinarizer
from utils.dataset_utility import EFRDataset, generate_tensor_utterances
from utils.models import EFRClass
from utils.models import save_model, load_model
import warnings
warnings.simplefilter('ignore')

# Hyper-parameters
EPOCHS = 1
SEEDS = [42, 123, 456, 843, 1296]
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LEARNING_RATE = 1e-05
MAX_LEN = 100

# DataLoader Hyper-paramaters
TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

TEST_PARAMS = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }







def readData(): 
    return pd.read_json('data/MELD_train_efr.json')


def create_dataframes(orginal_df, seed):
    train, test_validation = train_test_split(orginal_df, test_size=0.2, random_state=seed)
    validation, test = train_test_split(test_validation, test_size=0.5, random_state=seed)
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def set_default_seed(_seed):
    #_seed = SEEDS[0]
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

def replace_none_with_zero(lst):
    return [0 if x is None else x for x in lst]

def prepare_data(df, tokenizer):
    # Emotions One-hot Labeling
    df = one_hot_encoder(df, 'emotions', 'emotions_ids')
    # Speakers One-hot Labeling
    df = one_hot_encoder(df, 'speakers', 'speakers_ids')

    # Add padding to dialog
    max_n_dialogs = max(df['utterances'].apply(lambda x: len(x)))
    max_n_speakers = max(df['speakers'].apply(lambda x: len(x)))
    max_n_triggers = max(df['triggers'].apply(lambda x: len(x)))
    max_n_emotions = max(df['emotions'].apply(lambda x: len(x)))
    print(f"\nMaximum Number of Dialogs: {max_n_dialogs}")
    print(f"Maximum Number of Speakers: {max_n_speakers}")
    print(f"Maximum Number of Triggers: {max_n_triggers}")
    print(f"Maximum Number of Emotions: {max_n_emotions}\n")
    assert max_n_dialogs == max_n_speakers == max_n_triggers == max_n_emotions, "maximum number of features list are not equal"

    # Clean Data
    df['triggers'] = df['triggers'].apply(replace_none_with_zero)

    # Add paddings
    df[['utterances_input_ids','utternaces_attention_mask']] = df['utterances'].apply(lambda x: pd.Series(generate_tensor_utterances(x, tokenizer, MAX_LEN, max_n_dialogs),index = ['utterances_input_ids','utternaces_attention_mask']))
    df['trigger_ids'] = df['triggers'].apply(lambda x: torch.tensor(x + [0] * (max_n_dialogs - len(x)), dtype=torch.long))

    return df


def one_hot_encoder(df, column_name, new_column_name):
    mlb = MultiLabelBinarizer()
    binary_encoded = mlb.fit_transform(df[column_name])

    class_to_decimal = {label: i for i, label in enumerate(mlb.classes_)}
    df[new_column_name] = df[column_name].apply(lambda x: [class_to_decimal[label] for label in x])

    # Find the minum bits for classifying
    max_decimal = max(max(x) for x in df[new_column_name])
    min_bits = max(1, (max_decimal.bit_length() + 7) // 8) * 8
    print(f"At most {min_bits} bits needed for labeling {column_name}")

    df[new_column_name] = df[new_column_name].apply(lambda x: [format(decimal, f'0{min_bits}b') for decimal in x])
    return df


@cache
def predefined_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=True)
    pretrained_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    return tokenizer, pretrained_model

def save_dataframe(df):
    _folder = Path.cwd().joinpath("dataframes")
    if not _folder.exists():
        _folder.mkdir(parents=True)

    df_path = Path.joinpath(_folder, 'df_MELD_efr'+'.pkl')
    df.to_pickle(df_path)


def load_dataframe(_folder):
    df_path = Path.joinpath('dataframes/df_MELD_efr'+'.pkl')
    with open(df_path, 'rb') as file:
        df = pickle.load(file)
    return df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--load_df", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # Initialize the randomization
    seed = args.seed
    set_default_seed(seed)
    device = set_device()
    print("do train: ", args.do_train)

    # Retrieve pretrained model
    start = time.time()
    tokenizer, base_model = predefined_model()
    pre_model_frozen = copy.deepcopy(base_model)
    pre_model_unfrozen = copy.deepcopy(base_model)
    for param in pre_model_frozen.bert.parameters():
        param.requires_grad = False
    print(f"\nTime of retriving model and tokenizer: {time.time()-start:.1f}\n")

    # Inspecting Data
    start = time.time()
    #if args.load_df:
    #_df = load_dataframe()
    #else:
    _df = readData()
    _df = prepare_data(_df, tokenizer)
    save_dataframe(_df)
    print(f"Time of preparing data and saving it: {time.time()-start:.1f}\n")

    # Setup Datasets
    df_train, df_validation, df_test = create_dataframes(_df, seed)
    dataset_train = EFRDataset(df_train, tokenizer, MAX_LEN)
    dataset_validation = EFRDataset(df_validation, tokenizer, MAX_LEN)
    dataset_test = EFRDataset(df_test, tokenizer, MAX_LEN)

    # Setup DataLoaders
    loader_train = DataLoader(dataset_train, **TRAIN_PARAMS)
    loader_validation = DataLoader(dataset_validation, **TEST_PARAMS)
    loader_test = DataLoader(dataset_test, **TEST_PARAMS)

    #if args.do_train:
    model_frozen = EFRClass(pre_model_frozen, device)
    model_unfrozen = EFRClass(pre_model_unfrozen, device)
    pre_model_frozen.to(device)
    model_unfrozen.to(device)

    
    trainer = EFRTraining(loader_train, loader_validation, loader_test,  EPOCHS, device, seed, True)
    #trainer1 = EFRTraining(model_frozen, loader_train, loader_validation, loader_test, optimizer_frozen, EPOCHS, device, seed, False)
    optimizer_frozen = Adam(model_frozen.parameters(), lr=LEARNING_RATE)
    optimizer_unfrozen = Adam(model_unfrozen.parameters(), lr=LEARNING_RATE)
    trainer.train(model_unfrozen, optimizer_unfrozen)

    # elif args.do_eval:
    # loaded_tokenizer, loaded_model_frozen = load_model(seed,'frozen')
    # _, loaded_model_unfrozen = load_model(seed,'unfrozen')

    # trainer.test(loaded_model_frozen)



if __name__ == "__main__":
    main()