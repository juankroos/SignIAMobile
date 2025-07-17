import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
device = torch.device('cpu')
model.to(device)

# training ....finally
def training():
    print("Starting training... ahhh")
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
    
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch in progress_bar:
            # move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
        
            # forward pass 
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
         )
        
            loss = outputs.loss
            total_loss += loss.item()
        
            # backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
            progress_bar.set_postfix(loss=loss.item())
    
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
        # saving the best ones (model i mean...)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_bart_model.pt')
            print(f'best model saved with loss: {best_loss:.4f}')
'''