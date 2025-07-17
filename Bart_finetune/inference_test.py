from load_model import generate_summary, load_model
#path = ''
#model = load_model(path)
#text = ''
#print("Input:", text)
#print("Summary:", summary)

def infer(text,model):
    try:
        summary = gernerate_summary(text)
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None
    return summary



