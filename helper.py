import torch.nn as nn
from sklearn.metrics import classification_report
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class LSTMModel(nn.Module):
    def __init__(self, hidden_layers=None, dropout_rate=0.4, out_features=9):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_layers[0], num_layers=1, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        # dynamically build feedforward layers
        layers = []
        for in_dim, out_dim in zip(hidden_layers, hidden_layers[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self.relu)
            layers.append(self.dropout)

        # final classifier layer
        layers.append(nn.Linear(hidden_layers[-1], out_features))  # 9 classes

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])  # use last LSTM output with dropout
        return self.ffn(x)


class CNN1DModel(nn.Module):
    def __init__(self,
                 conv_config,  # list of (out_channels, kernel_size) tuples for conv layers
                 hidden_layers,  # list of hidden layer sizes for the fully connected classifier
                 dropout_rate,  # dropout rate applied after each layer
                 input_length,  # length of the input time series (using 151)
                 input_channels,  # number of input channels (just using 3)
                 out_features,  # number of output classes
                 pool_every=2):  # insert a MaxPool1d layer after every N conv layers
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        conv_layers = []
        in_channels = input_channels

        for i, (out_channels, kernel_size) in enumerate(conv_config):
            padding = kernel_size // 2
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            conv_layers.append(self.relu)

            if (i + 1) % pool_every == 0:
                conv_layers.append(nn.MaxPool1d(kernel_size=2))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        print("____________________update_3___________________________")

        # calculate flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_length)
            x = self.conv_layers(dummy_input)
            flatten_dim = x.view(1, -1).shape[1]

        # fully connected layers
        layers = []
        prev_dim = flatten_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.relu)
            layers.append(self.dropout)
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, out_features))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 3, 151)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.ffn(x)



class TransformerModel(nn.Module):
    def __init__(self, hidden_layers=[250, 250, 50], dropout_rate=0.4, seq_len=151, input_dim=3, out_features=9):
        super().__init__()
        self.d_model = hidden_layers[0]
        self.input_proj = nn.Linear(input_dim, self.d_model)

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.d_model))  # learnable positional encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=4 * self.d_model,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # dynamically build feedforward classifier
        layers = []
        for in_dim, out_dim in zip(hidden_layers, hidden_layers[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self.relu)
            layers.append(self.dropout)

        layers.append(nn.Linear(hidden_layers[-1], out_features))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.ffn(x)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, patience=10):
    print("pushed 1:44ipup")

    model.to(device)

    best_val_acc = 0.0
    patience_counter = 0

    # track loss and accuracy
    train_losses = []
    train_accuracies = []
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

        # compute average loss over the epoch
        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # evaluate train and validation accuracy
        train_acc = test_model(model, train_loader, device)
        val_acc = test_model(model, val_loader, device)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs. Best Val Acc: {best_val_acc:.4f}")
                break

    # plot accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs. Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_model(model, data_loader, device, print_output=False):
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

    acc = (np.array(all_preds) == np.array(all_labels)).mean()

    if print_output:
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

    return acc