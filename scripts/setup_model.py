import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def setup_model():
    # Check if model directory exists and is not empty
    if os.path.exists("model_save") and os.listdir("model_save"):
        print("Model directory already exists and contains files.")
        user_input = input("Do you want to overwrite existing model? (y/n): ")
        if user_input.lower() != 'y':
            print("Setup cancelled. Using existing model files.")
            return
        
    # Using a small BERT model for paraphrase detection
    model_name = "bert-base-uncased"  # or your specific model name
    
    print(f"Downloading model {model_name}...")
    
    # Download and save model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Saving model to model_save directory...")
    model.save_pretrained("model_save")
    tokenizer.save_pretrained("model_save")
    
    print("Model setup complete!")

if __name__ == "__main__":
    setup_model()
