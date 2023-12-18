import torch
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification


class EFRClass(torch.nn.Module):

    def __init__(self, pretrained_model, device):
        super(EFRClass, self).__init__()
        self.l1 = pretrained_model
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 4)
        self.device = device

    def forward(self, data):
        # Extract Data
        utterances_input_ids = data['utterances_input_ids'].to(self.device, dtype = torch.long)
        utternaces_attention_mask = data['utternaces_attention_mask'].to(self.device, dtype = torch.long)
        triggers = data['triggers'].to(self.device, dtype = torch.long)

        output = self.l1(utterances_input_ids, attention_mask=utternaces_attention_mask, labels=triggers)
        #output_conclusion = self.l1(input_ids=conclusion_input_ids, attention_mask=conclusion_attention_mask)
        #hidden_state = output_premise[0] + output_conclusion[0]         #shape (N, 256, 768)
        #hidden_state = torch.cat([hidden_state, stance_ids], dim=2)     #shape (N, 256, 769)
       #pooler = hidden_state[:, 0]                     #shape (N, 769)
        #pooler = self.pre_classifier(pooler)            #shape (769, 768)
        #pooler = torch.nn.Tanh()(pooler)
        #pooler = self.dropout(pooler)
        #output = self.classifier(pooler)
        return output
    

def save_model(model, seed, postfix):
    folder = Path.cwd().joinpath(f"models/{str(seed)}")
    if not folder.exists():
        folder.mkdir(parents=True)

    _path =f'models/{str(seed)}/emotion_model_{postfix}'
    torch.save(model, _path)


def save_tokenizer(tokenizer, seed):
    folder = Path.cwd().joinpath(f"models/{str(seed)}")
    if not folder.exists():
        folder.mkdir(parents=True)

    _path = f'models/{str(seed)}/emotion_tokenizer'
    tokenizer.save_vocabulary(_path)


def load_model(seed, postfix):
    return torch.load(f'models/{str(seed)}/emotion_model_{postfix}')


def load_tokenizer(seed):  
    return BertTokenizer.from_pretrained(f'models/{str(seed)}/emotion_tokenizer') 