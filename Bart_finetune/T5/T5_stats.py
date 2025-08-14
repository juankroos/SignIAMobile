import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset
import evaluate
import numpy as np
from tqdm import tqdm

# Charger les métriques BLEU et ROUGE
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

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


def generate_corrections(model, tokenizer, texts, device, max_length=128):
    """Generate corrections for a list of texts."""
    model.eval()
    corrections = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Generating corrections"):
            # Ajouter le préfixe T5
            input_text = f"correct: {text}"
            
            # Tokenizer l'entrée
            inputs = tokenizer(input_text, max_length=max_length, truncation=True, 
                             padding='max_length', return_tensors="pt").to(device)
            
            # Générer la correction
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            # Décoder la sortie
            correction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrections.append(correction)
    
    return corrections


def compute_metrics(predictions, references):
    """Compute BLEU and ROUGE metrics."""
    
    # Calcul BLEU
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        # BLEU attend des listes de tokens
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]  # BLEU attend une liste de références
        
        try:
            bleu_score = bleu_metric.compute(predictions=[pred_tokens], references=[ref_tokens])
            bleu_scores.append(bleu_score['bleu'])
        except:
            bleu_scores.append(0.0)
    
    avg_bleu = np.mean(bleu_scores)
    
    # Calcul ROUGE
    try:
        rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    except:
        rouge_scores = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'rougeLsum': 0.0
        }
    
    return {
        'bleu': avg_bleu,
        'rouge1': rouge_scores.get('rouge1', 0.0),
        'rouge2': rouge_scores.get('rouge2', 0.0),
        'rougeL': rouge_scores.get('rougeL', 0.0),
        'rougeLsum': rouge_scores.get('rougeLsum', 0.0)
    }


def main():
    DATA_PATH = 'phrases_new.txt'
    model_name = 't5-small'
    max_length = 128
    batch_size = 4
    epochs = 5
    learning_rate = 5e-5

    # Chargement des données
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

    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Preprocess Data
    train_encodings = preprocess_data(tokenizer, train_texts, train_labels, max_length)
    val_encodings = preprocess_data(tokenizer, val_texts, val_labels, max_length)

    # DataLoader
    train_dataset = GrammarCorrectionDataset(train_encodings)
    val_dataset = GrammarCorrectionDataset(val_encodings)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = get_device()
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in tqdm(data_loader, desc="Training"):
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

    # Evaluation function
    def evaluate(model, data_loader, device):
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    print("=" * 50)
    print("STARTING T5 FINE-TUNING")
    print("=" * 50)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        val_loss = evaluate(model, val_loader, device)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(" Saving best model...")
            model_dir = './models/t5-text-corrector-best'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    print("\n" + "=" * 50)
    print("FINE-TUNING COMPLETED - COMPUTING METRICS")
    print("=" * 50)

    # Charger le meilleur modèle pour l'évaluation
    best_model = T5ForConditionalGeneration.from_pretrained('./models/t5-text-corrector-best')
    best_model.to(device)

    # Générer les corrections sur l'ensemble de validation
    print("\n Generating corrections on validation set...")
    predictions = generate_corrections(best_model, tokenizer, val_texts, device, max_length)

    # Calculer les métriques
    print(" Computing BLEU and ROUGE metrics...")
    metrics = compute_metrics(predictions, val_labels)

    # Afficher les résultats
    print("\n" + "=" * 50)
    print(" EVALUATION METRICS")
    print("=" * 50)
    print(f" BLEU Score: {metrics['bleu']:.4f}")
    print(f" ROUGE-1: {metrics['rouge1']:.4f}")
    print(f" ROUGE-2: {metrics['rouge2']:.4f}")
    print(f" ROUGE-L: {metrics['rougeL']:.4f}")
    print(f" ROUGE-Lsum: {metrics['rougeLsum']:.4f}")

    # Sauvegarder les métriques
    metrics_file = './models/t5-text-corrector-best/metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("T5 Grammar Correction - Evaluation Metrics\n")
        f.write("=" * 45 + "\n")
        f.write(f"BLEU Score: {metrics['bleu']:.4f}\n")
        f.write(f"ROUGE-1: {metrics['rouge1']:.4f}\n")
        f.write(f"ROUGE-2: {metrics['rouge2']:.4f}\n")
        f.write(f"ROUGE-L: {metrics['rougeL']:.4f}\n")
        f.write(f"ROUGE-Lsum: {metrics['rougeLsum']:.4f}\n")
    
    print(f"\n Metrics saved to: {metrics_file}")

    # Montrer quelques exemples
    print("\n" + "=" * 50)
    print(" SAMPLE CORRECTIONS")
    print("=" * 50)
    for i in range(min(5, len(val_texts))):
        print(f"\nExample {i+1}:")
        print(f"Input:      {val_texts[i]}")
        print(f"Target:     {val_labels[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 30)

    # Sauvegarder le modèle final
    final_model_dir = './models/t5-text-corrector'
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"\n Final model saved to: {final_model_dir}")
    print(" Training and evaluation completed successfully!")


if __name__ == '__main__':
    main()