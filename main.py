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
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from functools import cache
import time
import argparse
import glob
import logging
from train_val import EFRTraining
from sklearn.preprocessing import MultiLabelBinarizer

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




class EFRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.utterances = dataframe.utterances
        self.triggers = dataframe.triggers
        self.speakers = dataframe.speakers
        self.max_len = max_len
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        utterance = self.utterances[index]
        trigger = self.triggers[index]

        utt_tokenized = list(map(lambda x: self.tokenizer.encode_plus(x, add_special_tokens=True, truncation=True), utterance))     
        #utt_encodings = self.tokenize55r.encode_plus(utterance, add_special_tokens=True, return_tensors="pt")
        # print(utterance)
        print("TTTTTTTTTTTTTTTTTTTTTTTTT")
        #print(utt_tokenized) 

        return {
            'utterances_tokenized': torch.tensor(utt_tokenized[0]['input_ids'], dtype=torch.long),
            'triggers': torch.tensor(trigger, dtype=torch.long)
        }


def readData(): 
    return pd.read_json('data/MELD_train_efr.json')


def create_dataframes(orginal_df, seed):
    train, test_validation = train_test_split(orginal_df, test_size=0.2, random_state=seed)
    validation, test = train_test_split(test_validation, test_size=0.5, random_state=seed)
    return train, validation, test


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


def prepare_data(df):
    # Emotions One-hot Labeling
    df = one_hot_encoder(df, 'emotions', 'emotions_ids')
    # Speakers One-hot Labeling
    df = one_hot_encoder(df, 'speakers', 'speakers_ids')

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # Initialize the randomization
    seed = args.seed
    set_default_seed(seed)

    # Set Device
    device = set_device()
    print("do train: ", args.do_train)

    # Retrieve pretrained model
    start = time.time()
    tokenizer, model = predefined_model()
    print(f"Retriving Model and Tokenizer time: {time.time()-start}")

    # Inspecting Data
    _df = readData()
    
    max_n_dialogs = max(_df['utterances'].apply(lambda x: len(x)))
    max_n_speakers = max(_df['speakers'].apply(lambda x: len(x)))
    max_n_triggers = max(_df['triggers'].apply(lambda x: len(x)))
    max_n_emotions = max(_df['emotions'].apply(lambda x: len(x)))
    print(f"\nMaximum Number of Dialogs: {max_n_dialogs}")
    print(f"Maximum Number of Speakers: {max_n_speakers}")
    print(f"Maximum Number of Triggers: {max_n_triggers}")
    print(f"Maximum Number of Emotions: {max_n_emotions}\n")

    _df = prepare_data(_df)

    # Setup Datasets
    df_train, df_validation, df_test = create_dataframes(_df, seed)
    dataset_train = EFRDataset(df_train, tokenizer, MAX_LEN)
    #dataset_validation = EFRDataset(df_validation, tokenizer, MAX_LEN)
    #dataset_test = EFRDataset(df_test, tokenizer, MAX_LEN)

    # Setup DataLoaders
    loader_train = DataLoader(dataset_train, **TRAIN_PARAMS)
    # loader_validation = DataLoader(dataset_validation, **TEST_PARAMS)
    # loader_test = DataLoader(dataset_test, **TEST_PARAMS)

    #if args.do_train:
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # trainer = EFRTraining(model, loader_train, optimizer, EPOCHS, device).train()


    # elif args.do_eval:
    #     # Load



    

    


    print(_df['speakers'][0])
    print(_df['utterances'][0])
    print(_df['triggers'][0])
    print(_df.keys())
    print(len(_df))
    prepare_data()


if __name__ == "__main__":
    main()