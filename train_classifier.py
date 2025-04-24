import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Define a custom dataset class
class MathProblemDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_length):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }

file_dir = "./labels"

# Function to load data from JSON files
def load_data(file_paths):
    questions = []
    labels = []
    for file_path, label in file_paths:
        file_path = os.path.join(file_dir, file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                questions.append(item['question'])
                labels.append(label)
    return questions, labels


# Define file paths and corresponding labels
file_paths = [
    ('easy.jsonl', 'easy'),
    ('hard.jsonl', 'hard'),
    ('very_hard.jsonl', 'very hard'),
]

# Load data
questions, labels = load_data(file_paths)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training and validation sets
train_questions, val_questions, train_labels, val_labels = train_test_split(
    questions, encoded_labels, test_size=0.1, random_state=42
)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir = "/scratch/yl9038/.cache")

# Create datasets and data loaders
max_length = 128
train_dataset = MathProblemDataset(train_questions, train_labels, tokenizer, max_length)
val_dataset = MathProblemDataset(val_questions, val_labels, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")

# Save the model and label encoder
model.save_pretrained('bert_math_classifier')
tokenizer.save_pretrained('bert_math_classifier')
with open('label_encoder.json', 'w') as f:
    json.dump(label_encoder.classes_.tolist(), f)
