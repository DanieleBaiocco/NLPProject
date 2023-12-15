import torch
from torch.utils.data import Dataset


def add_padding(utterance_list, n_pad, max_len):
    for _ in range(n_pad):
        utterance_list.append(generate_dict_padding(max_len))
    return utterance_list


def generate_dict_padding(max_len):
    return {
        'input_ids': torch.tensor([0] * max_len),
        'attention_mask': torch.tensor([0] * max_len)
    }     


def generate_tensor_utterances(utterance, tokenizer, max_len, max_n_dialogs):
    utterance_list = []
    for utt in utterance:
        tokenized_utt = tokenizer.encode_plus(utt, add_special_tokens=True, truncation=True, max_length=max_len, pad_to_max_length=True)
        utterance_list.append({
                                'input_ids':torch.tensor(tokenized_utt['input_ids'], dtype=torch.long),
                                'attention_mask':torch.tensor(tokenized_utt['attention_mask'], dtype=torch.long)
                                })
    _n_pad = max_n_dialogs-len(utterance_list)
    utterance_list = add_padding(utterance_list, _n_pad, max_len)
    return utterance_list


class EFRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        #self.utterances = dataframe.utterances
        self.tokenized_utterances = dataframe.tokenized_utterances
        self.triggers = dataframe.trigger_ids
        self.speakers = dataframe.speakers
        self.max_len = max_len
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        tokenized_utterances = self.tokenized_utterances[index]
        triggers = self.triggers[index]
        #trigger = self.triggers[index]
        #utt_tokenized = list(map(lambda x: self.tokenizer.encode_plus(x, add_special_tokens=True, truncation=True, max_length=self.max_len, pad_to_max_length=True), utterance))
                            
        #utt_encodings = self.tokenize55r.encode_plus(utterance, add_special_tokens=True, return_tensors="pt")
        # print(utterance)
        print("TTTTTTTTTTTTTTTTTTTTTTTTT")
        #print(utt_tokenized) 

        return {
            'utterances': tokenized_utterances,                      # Dict k:(input_ids, attention_mask)  tensor
            'triggers': triggers    # List tensor
        }