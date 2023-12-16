import torch
from torch.utils.data import Dataset


def add_padding(_list, n_pad, max_len):
    for _ in range(n_pad):
        _list.append([0] * max_len)
    return _list


def generate_tensor_utterances(utterance, tokenizer, max_len, max_n_dialogs):
    input_ids= []
    attention_mask = []
    for utt in utterance:
        tokenized_utt = tokenizer.encode_plus(utt, add_special_tokens=True, truncation=True, max_length=max_len, padding='max_length')
        input_ids.append(tokenized_utt['input_ids'])
        attention_mask.append(tokenized_utt['attention_mask'])

    _n_pad = max_n_dialogs-len(input_ids)
    input_ids = add_padding(input_ids, _n_pad, max_len)
    attention_mask  = add_padding(attention_mask, _n_pad, max_len)
    return input_ids, attention_mask


class EFRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        #self.utterances = dataframe.utterances
        self.utterances_input_ids = dataframe.utterances_input_ids
        self.utternaces_attention_mask = dataframe.utternaces_attention_mask
        self.triggers = dataframe.trigger_ids
        self.speakers = dataframe.speakers
        self.max_len = max_len
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        utterances_input_ids = self.utterances_input_ids[index]
        utternaces_attention_mask = self.utternaces_attention_mask[index]
        _triggers = self.triggers[index]

        return {
            'utterances_input_ids': torch.tensor(utterances_input_ids[0], dtype=torch.long),           # Dict k:(input_ids, attention_mask) 
            'utternaces_attention_mask': torch.tensor(utternaces_attention_mask[0],dtype=torch.long),  # Dict k:(input_ids, attention_mask)  
            'triggers': _triggers[0]                                                                   # List 
        }