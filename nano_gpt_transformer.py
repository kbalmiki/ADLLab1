import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
import re
from matplotlib import pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# ----------------------- Preprocessing -----------------------

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def preprocess_pandas(data):
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'\d', '', regex=True)

    stop_words = set(stopwords.words('english'))
    data['tokens'] = data['Sentence'].apply(lambda x: [w for w in simple_tokenize(x) if w not in stop_words])
    return data

# ----------------------- Vocabulary -----------------------

def build_vocab(token_lists, min_freq=1):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode_sentence(tokens, vocab, max_len=30):
    encoded = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(encoded) < max_len:
        encoded += [vocab['<PAD>']] * (max_len - len(encoded))
    return encoded[:max_len]

# ----------------------- Transformer Model -----------------------

class GPTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, num_classes=2, max_len=30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)  # Mean pooling
        return self.fc(x)

# ----------------------- Main -----------------------

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv("amazon_cells_labelled_LARGE_25K.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data = preprocess_pandas(data)

    # Build vocabulary and encode sentences
    vocab = build_vocab(data['tokens'])
    max_len = 30
    data['input_ids'] = data['tokens'].apply(lambda x: encode_sentence(x, vocab, max_len))

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        list(data['input_ids']),
        list(data['Class']),
        test_size=0.1,
        random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Dataloaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # Model, optimizer, loss
    model = GPTClassifier(vocab_size=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Plot losses
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    # Evaluation
    def evaluate(model, dataloader):
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1)
                preds.extend(pred.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
        return labels_all, preds

    y_true, y_pred = evaluate(model, val_loader)

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
