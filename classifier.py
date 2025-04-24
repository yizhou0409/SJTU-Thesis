import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model and tokenizer
model_path = 'bert_math_classifier'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load the label encoder (list of label classes)
with open('label_encoder.json', 'r') as f:
    label_classes = json.load(f)

def classify(question, max_length=512):
    # Tokenize the input question
    encoding = tokenizer(
        question,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_index = torch.argmax(logits, dim=1).item()

    # Map the numerical prediction back to the label
    return label_classes[predicted_index]

# Example usage:
question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?"
predicted_label = classify(question)
print("Predicted Difficulty:", predicted_label)