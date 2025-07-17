import os
import pandas as pd
from Bart_finetune.Finetuning import MAX_DECODER_SEQUENCE_LENGTH, MAX_ENCODER_SEQUENCE_LENGTH
from torch.utils.data import Dataset, DataLoader



class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data.iloc[idx]['input_text']
        tgt = self.data.iloc[idx]['output_text']
        
        encoder_inputs = self.tokenizer(
            src,
            max_length=MAX_ENCODER_SEQUENCE_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        decoder_inputs = self.tokenizer(
            tgt,
            max_length=MAX_DECODER_SEQUENCE_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = decoder_inputs['input_ids'].squeeze().clone()
        # replace padding tokens with -100 for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_inputs['input_ids'].squeeze(),
            'labels': labels
        }