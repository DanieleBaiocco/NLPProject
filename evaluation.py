from utils.models import save_model, load_model, save_tokenizer
from sklearn.metrics import f1_score, classification_report, accuracy_score
from pathlib import Path
import pickle

SEEDS = [42, 123, 432, 817, 1432]

def evaluate(tester):

    models = {}
    # 10 Models will be retrieved
    for _seed in SEEDS:
        for model_type in ['frozen', 'unfrozen']:
            models[f'emotion_{_seed}_{model_type}'] = load_model(_seed, model_type)


    metrics = {}
    for _model in models:
        print(f"Testing Model: {_model}")
        targets, final_outputs = tester.test()

        metrics[f'model'+'_f1_score'] = f1_score(targets, final_outputs, average=None)
        metrics[f'model'+'_macro'] = f1_score(targets, final_outputs, average='macro')
        metrics[f'model'+'_accuracy'] = accuracy_score(targets, final_outputs)
        metrics[f'model'+'_report'] = classification_report(targets, final_outputs)

        print(metrics[f'model'+'_report'] )

    save_metrics(metrics)   


def save_metrics(metrics):
    folder_metrics = Path.cwd().joinpath(f"metrics")
    if not folder_metrics.exists():
        folder_metrics.mkdir(parents=True)

    with open(f'emotion_metrics.pkl', 'wb') as file:
        pickle.dump(metrics, file)


def load_metrics():
    with open(f'metrics/emotion_metrics.pkl', 'rb') as file:
       loaded_metrics = pickle.load(file)

    return loaded_metrics