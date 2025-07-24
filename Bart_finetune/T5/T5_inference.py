import torch
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer


def get_device():
    """Return the available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def correct_text(model, tokenizer, texts, max_length=128, device="cpu"):
    """Generate corrected text for a list of input sentences."""
    # Add T5 prefix for grammar correction
    inputs = [f"correct: {text}" for text in texts]

    # Tokenize inputs
    encodings = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    # Move to device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Generate outputs
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,  # Beam search for better results
            early_stopping=True
        )

    # Decode outputs
    corrected_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return corrected_texts


def main(model_dir, input_texts=None):
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)

    # Set device
    device = get_device()
    model.to(device)

    # Default input texts if none provided
    if input_texts is None:
        input_texts = [
            "garden flower water grow beautiful",
            "car key lock inside wait",
            "she want book new black today ",
            "coffee too hot",
            "bicycle ride hill fast brake",
            "concert favorite song sing loud"
        ]

    # Perform inference
    corrected_texts = correct_text(model, tokenizer, input_texts, device=device)

    # Print results
    for original, corrected in zip(input_texts, corrected_texts):
        print(f"Original: {original}")
        print(f"Corrected: {corrected}")
        print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform text correction using a fine-tuned T5 model.")
    parser.add_argument('--model_dir', type=str, default='./models/t5-text-corrector',
                        help='Path to the directory containing the fine-tuned model and tokenizer')
    parser.add_argument('--input_texts', type=str, nargs='+',
                        help='List of input sentences to correct (e.g., "i go school" "you eat lunch")')
    args = parser.parse_args()

    main(args.model_dir, args.input_texts)