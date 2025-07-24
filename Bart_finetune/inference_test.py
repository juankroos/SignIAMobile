import torch
from transformers import BartTokenizer, BartForConditionalGeneration


MAX_ENCODER_SEQUENCE_LENGTH = 1024
MAX_GENERATION_LENGTH = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model and tokenizer initialization
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.load_state_dict(torch.load('best_bart_model.pt', map_location=device))
model.to(device)
model.eval()

# #inference function
def generate_summary(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        max_length=MAX_ENCODER_SEQUENCE_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=MAX_GENERATION_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# try 
test_input = "wash face bathroom"
summary = generate_summary(test_input, model, tokenizer, device)
print("Input:", test_input)
print("Generated Summary:", summary)