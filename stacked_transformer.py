import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')

# -----------------------------------------
# Preprocessing Function
# -----------------------------------------
def preprocess_text(data):
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'\d', '', regex=True)

    stop_words = set(stopwords.words('english'))

    def clean(sentence):
        words = re.findall(r'\b\w+\b', sentence)
        return " ".join([word for word in words if word not in stop_words])

    data['Sentence'] = data['Sentence'].apply(clean)
    return data

# -----------------------------------------
# Transformer Model (Stacked)
# -----------------------------------------
class StackedTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=128, num_layers=2):
        super(StackedTransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.input_proj(x)  # (batch_size, hidden_dim)
        x = x.unsqueeze(1)      # (batch_size, seq_len=1, hidden_dim)
        x = x.permute(1, 0, 2)  # (seq_len=1, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)        # (batch_size, hidden_dim)
        return self.fc(x)

# -----------------------------------------
# Main Block
# -----------------------------------------
if __name__ == "__main__":
    # Load and preprocess
    data = pd.read_csv("amazon_cells_labelled_LARGE_25K.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data = preprocess_text(data)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data['Sentence']).todense()
    y = data['Class'].values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # Model
    input_size = X_train.shape[1]
    model = StackedTransformerClassifier(input_dim=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    train_losses = []
    val_losses = []
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Evaluation
    def evaluate(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_labels, all_preds

    labels, preds = evaluate(model, val_loader)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["Negative", "Positive"])

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n", report)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # Plot Losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
