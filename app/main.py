import os
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create FastAPI app
app = FastAPI(
    title="Paraphrase Classifier API",
    description="API for classifying sentence pairs as paraphrases or non-paraphrases",
    version="1.0.0"
)

# Load model and tokenizer
model_dir = os.getenv("MODEL_DIR", "model_save")
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.get("/info")
def info():
    return {
        "name": "paraphrase-classifier",
        "description": "BERT-based paraphrase classification API",
        "model": model.config.name_or_path
    }

@app.get("/classify")
async def classify_paraphrase(sentence1: str, sentence2: str) -> Dict:
    # Tokenize the sentence pair
    inputs = tokenizer(
        sentence1,
        sentence2,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        prediction = predictions.argmax().item()
        confidence = predictions[0][prediction].item()

    # Return classification result
    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "is_paraphrase": bool(prediction),
        "confidence": round(confidence, 4)
    }
