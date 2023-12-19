from utils.models import save_model, load_model, save_tokenizer
from sklearn.metrics import f1_score, classification_report, accuracy_score
from pathlib import Path
import pickle
import pandas as pd

#SEEDS = [42, 123, 432, 817, 1432]
SEEDS = [42]

def evaluate(tester):

    models = {}
    # 10 Models will be retrieved
    for _seed in SEEDS:
        for model_type in ['frozen', 'unfrozen']:
            models[f'emotion_{_seed}_{model_type}'] = load_model(_seed, model_type)


    metrics = {}
    for _m_name, _m in models.items():
        print(f"Testing Model: {_m_name}")
        targets, final_outputs = tester.test(_m)

        metrics[f'{_m_name}_f1_score'] = f1_score(targets, final_outputs, average=None)
        metrics[f'{_m_name}_macro'] = f1_score(targets, final_outputs, average='macro')
        metrics[f'{_m_name}_accuracy'] = accuracy_score(targets, final_outputs)
        metrics[f'{_m_name}_report'] = classification_report(targets, final_outputs)

        print(metrics[f'{_m_name}_report'] )

    save_metrics(metrics)  

    # Presenting Table
    _df = pd.DataFrame()
    # Build columns 
    other_cols = ['f1_score', 'macro avergage', 'accuracy']
    _df['model']  = list(models.keys())

    for k, _ in models.items():
       _df.loc[_df['model'] == k, other_cols] = [metrics[f'{k}_f1_score'][0], metrics[f'{k}_macro'], metrics[f'{k}_accuracy']]

    print(_df)

    

def save_metrics(metrics):
    folder_metrics = Path.cwd().joinpath(f"metrics")
    if not folder_metrics.exists():
        folder_metrics.mkdir(parents=True)

    with open(f'metrics/emotion_metrics.pkl', 'wb') as file:
        pickle.dump(metrics, file)


def load_metrics():
    with open(f'metrics/emotion_metrics.pkl', 'rb') as file:
       loaded_metrics = pickle.load(file)

    return loaded_metrics