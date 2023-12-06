import csv
import re
import ast

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

def process_row(row):
    id_value = int(row[0])
    names = ast.literal_eval(row[1])
    emotions = ast.literal_eval(row[2])
    sentences = ast.literal_eval(row[3])
    scores = ast.literal_eval(row[4])

    result_rows = []

    for name, emotion, sentence, score in zip(names, emotions, sentences, scores):
        result_rows.append([id_value, name, emotion, sentence, score])

    return result_rows


def process_csv(input, output):
    # Read the CSV file and process the data
    with open(input, 'r', newline='', encoding='utf-8') as input_file, \
            open(output, 'w', newline='', encoding='utf-8') as output_file:

        csv_reader = csv.reader(input_file, delimiter=',')
        csv_writer = csv.writer(output_file, delimiter=',')

        # Process and write the header row
        header_row = next(csv_reader)
        csv_writer.writerow(header_row)

        for row in csv_reader:
            try:
                processed_rows = process_row(row)
                for processed_row in processed_rows:
                    csv_writer.writerow(processed_row)
            except:
                pass


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
df.iloc[:, 1:] = df.iloc[:,1:].map(replace_nan_with_zero)
df["episode"] = df["episode"].apply(lambda x: re.findall("\d+", x)[0])

df.rename(columns={"episode": "Dialogue_Id",
                   "utterances": "Utterance",
                   "speakers": "Speaker",
                   "emotions": "Emotion_name",
                   "triggers": "Annotate(0/1)"},
          inplace=True)

#df.to_csv('output.csv', index=False)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv('CSVs/meld-fr_partial_train_unprocessed.csv', index=False)
test_data.to_csv('CSVs/meld-fr_partial_test_unprocessed.csv', index=False)

input_csv_train_path = 'CSVs/meld-fr_partial_train_unprocessed.csv'
output_csv_train_path = 'CSVs/meld-fr_partial_train.csv'

input_csv_test_path = 'CSVs/meld-fr_partial_test_unprocessed.csv'
output_csv_test_path = 'CSVs/meld-fr_partial_test.csv'

process_csv(input_csv_train_path, output_csv_train_path)
process_csv(input_csv_test_path, output_csv_test_path)