import torch.nn as nn
from sklearn.metrics import classification_report
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=250, num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(250, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 9)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(lstm_out[:, -1, :])
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, patience=10):
    model.to(device)

    best_val_acc = 0.0
    patience_counter = 0

    # lists to track metrics
    train_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        val_acc = test_model(model, val_loader, device, return_acc=True)

        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs. Best Val Acc: {best_val_acc:.4f}")
                break

    # plot after training
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def test_model(model, data_loader, device, return_acc=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    if return_acc:
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        return acc
    else:
        # Classification report
        print("Classification Report:\n", classification_report(all_labels, all_preds))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[f"Pred {i}" for i in range(cm.shape[0])],
                    yticklabels=[f"True {i}" for i in range(cm.shape[0])])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
