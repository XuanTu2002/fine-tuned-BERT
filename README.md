# BERT Paraphrase Classifier API

Live API: https://fine-tuned-bert.onrender.com/

## Endpoints

### Health Check
```bash
GET https://fine-tuned-bert.onrender.com/
```

### API Information
```bash
GET https://fine-tuned-bert.onrender.com/info
```

### Classify Paraphrases
```bash
GET https://fine-tuned-bert.onrender.com/classify?sentence1={text1}&sentence2={text2}
```

Example:
```bash
curl -X GET "https://fine-tuned-bert.onrender.com/classify?sentence1=The%20cat%20is%20on%20the%20mat&sentence2=A%20cat%20sits%20on%20the%20mat"
```

Response format:
```json
{
    "sentence1": "The cat is on the mat",
    "sentence2": "A cat sits on the mat",
    "is_paraphrase": true,
    "confidence": 0.9876
}
```
