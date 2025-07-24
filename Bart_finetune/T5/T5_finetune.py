import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset



# utility functions
def get_device():
    """Return the available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data(tokenizer, texts, labels, max_length):
    """Tokenize input texts and labels for T5."""
    # Add T5 prefix for grammar correction task
    inputs = [f"correct: {text}" for text in texts]
    input_encodings = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length',
                                return_tensors="pt")
    target_encodings = tokenizer(labels, max_length=max_length, truncation=True, padding='max_length',
                                 return_tensors="pt")

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }


class GrammarCorrectionDataset(Dataset):
    """Custom Dataset for grammar correction."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def main():
    DATA_PATH = 'phrases_new.txt'
    model_name = 't5-small'
    max_length = 128
    batch_size = 4  #
    epochs = 5
    learning_rate = 5e-5

    incorrect_texts = []
    correct_texts = []
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        for line in file:

            incorrect, correct = line.strip().split(',')
            incorrect_texts.append(incorrect.strip())
            correct_texts.append(correct.strip())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        incorrect_texts, correct_texts, test_size=0.2, random_state=42
    )

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # preprocess Data
    train_encodings = preprocess_data(tokenizer, train_texts, train_labels, max_length)
    val_encodings = preprocess_data(tokenizer, val_texts, val_labels, max_length)

    # dataLoader
    train_dataset = GrammarCorrectionDataset(train_encodings)
    val_dataset = GrammarCorrectionDataset(val_encodings)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model
    device = get_device()
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # training function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # evaluation function
    def evaluate(model, data_loader, device):
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    # training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

    # save model
    model_dir = './models/t5-text-corrector'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == '__main__':
    main()