import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from Bart_finetune.Dataset import TextDataset

#hyperparameters
BATCH_SIZE = 6
EPOCHS = 8
MAX_ENCODER_SEQUENCE_LENGTH = 1024
MAX_DECODER_SEQUENCE_LENGTH = 512
MAX_GENERATION_LENGTH = 256
LEARNING_RATE = 5e-5


#load data
data_path = 'phrases.txt'
df = pd.read_csv(data_path, header=None, names=['input_text', 'output_text'], encoding='utf-8')
print(df.head())

'''
# and thre we make some dataset for tokenization and batching... a class
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
'''

# initialize the model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')


#creating the dataset loader
dataset = TextDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# chosing the device for training cpu, gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
device = torch.device('cpu')


# optimizer and sheduler here is AdamW and linear scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = int(total_steps * 0.1),
    num_training_steps = total_steps
)

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
                input_ids = input_ids,
                attention_mask = attention_mask,
                decoder_input_ids = decoder_input_ids,
                labels = labels
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
# to load any model using path
def load_model(model):
    model.load_state_dict(torch.load(model))
    model.eval()
'''

'''
# load the best model one for inference
model.load_state_dict(torch.load('best_bart_model.pt'))
model.eval()

# generating summaries
def generate_summary(text,model1):
    inputs = tokenizer(
        text,
        max_length=MAX_ENCODER_SEQUENCE_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    summary_ids = model1.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=MAX_GENERATION_LENGTH,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# finnaly testing .... testing !!!
test_input = "something to test all that..."
summary = generate_summary(test_input)
print("Input:", test_input)
print("Generated Summary:", summary)
'''