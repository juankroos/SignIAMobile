from Model_video import Extractor
### function to load the model
# load the best model one for inference
model = model.load_state_dict(torch.load('best_bart_model.pt'))
model.eval()

#text comming from the video model inference
#text = text_generated
# generating summaries
def generate_summary(text):
    inputs = tokenizer(
        text,
        max_length=MAX_ENCODER_SEQUENCE_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=MAX_GENERATION_LENGTH,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
'''
# finnaly testing .... testing !!!
test_input = "something to test all that..."
summary = generate_summary(test_input)
print("Input:", test_input)
print("Generated Summary:", summary)
'''