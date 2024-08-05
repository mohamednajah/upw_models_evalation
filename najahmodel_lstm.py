import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

df = pd.read_csv('cleaned_data.csv')  
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Beschreibung'], df['target'], test_size=0.2, random_state=42
)

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx].clone().detach(), dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(lstm_out)
        return out

def tokenize(texts, tokenizer):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)

def evaluate_model(model, train_loader, test_loader, criterion, optimizer, device):
    model.to(device)
    model.train()

    for epoch in range(1): 
        for batch in train_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()

    correct, total = 0, 0
    preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='weighted')
    
    if len(set(all_labels)) == 2:
        outputs = torch.softmax(torch.tensor(preds), dim=1).numpy()[:, 1]
        auc = roc_auc_score(all_labels, outputs)
    else:
        auc = 'N/A'

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auc == 'N/A':
        print("ROC-AUC: N/A")
    else:
        print(f"ROC-AUC: {auc:.4f}")

embedding_dim = 100
hidden_dim = 128
output_dim = len(set(train_labels))  

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenize(train_texts, tokenizer)
test_encodings = tokenize(test_texts, tokenizer)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

model = LSTMModel(vocab_size=len(tokenizer), embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluate_model(model, train_loader, test_loader, criterion, optimizer, device)
