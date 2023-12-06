import csv
from json.decoder import NaN

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import json
from pathlib import Path
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

dataset_name = "MELD_train_efr.json"

print(f"Current work directory: {Path.cwd()}")
dataset_folder = Path.cwd().joinpath("Datasets")

dataset_path = dataset_folder.joinpath(dataset_name)

f = open(dataset_path)

data = json.load(f)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Define a lambda function to replace NaN with 0 in arrays
replace_nan_with_zero = lambda x: [0 if pd.isna(val) else val for val in x]

# Apply the lambda function using applymap
df = df.map(replace_nan_with_zero)

#df.to_csv('output.csv', index=False)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data.to_csv('CSVs/train.csv', index=False)
test_data.to_csv('CSVs/test.csv', index=False)