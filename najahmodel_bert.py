from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

df = pd.read_csv('cleaned_data.csv') 
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Beschreibung'], df['target'], test_size=0.2, random_state=42
)

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

def tokenize(texts, tokenizer):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)

def evaluate_model(model_name, tokenizer, train_encodings, train_labels, test_encodings, test_labels):
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Evaluation results for {model_name}: {eval_results}")
    
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    labels = test_labels.values

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    if len(set(labels)) == 2:
        auc = roc_auc_score(labels, logits[:, 1])
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

    return eval_results, accuracy, precision, recall, f1, auc

tokenizers = {'bert-base-uncased': BertTokenizer.from_pretrained('bert-base-uncased')}
models = ['bert-base-uncased']

for model_name in models:
    tokenizer = tokenizers[model_name]
    train_encodings = tokenize(train_texts, tokenizer)
    test_encodings = tokenize(test_texts, tokenizer)
    evaluate_model(model_name, tokenizer, train_encodings, train_labels, test_encodings, test_labels)
