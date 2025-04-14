import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from nltk.corpus import stopwords
import re
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from collections import Counter

# Download NLTK stopwords if not already
import nltk
nltk.download('stopwords')

# Tokenizer function
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        tokens = simple_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords.words("english")]
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        return torch.tensor(ids[:self.max_len])

    def __getitem__(self, idx):
        x = self.encode(self.texts[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

# Collate function to pad batches
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True)
    return inputs_padded, torch.stack(targets)

# Transformer-based Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, num_classes=2, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.rand(1, max_len, embed_dim))  # simple learnable positional encoding

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = x.permute(1, 0, 2)  # seq_len, batch, embed_dim
        x = self.transformer(x)
        x = x.mean(dim=0)  # average pooling
        return self.fc(x)

# Load and preprocess data
df = pd.read_csv("amazon_cells_labelled_LARGE_25K.txt", delimiter='\t', header=None)
df.columns = ['Sentence', 'Class']

# Build vocabulary
all_tokens = []
for text in df['Sentence']:
    tokens = simple_tokenize(text)
    filtered = [t for t in tokens if t not in stopwords.words("english")]
    all_tokens.extend(filtered)

token_freq = Counter(all_tokens)
vocab = {word: i+2 for i, (word, _) in enumerate(token_freq.most_common(10000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Sentence'], df['Class'], test_size=0.1, random_state=42)

# Dataset and Dataloader
train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), vocab)
val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(vocab_size=len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses, val_losses = [], []
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

# Evaluation
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
conf_mat = confusion_matrix(all_labels, all_preds)

print(f"\nFinal Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:\n", report)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Negative", "Positive"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot training and validation loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
