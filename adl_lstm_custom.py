from preprocessing import load_blockwise_sequences, load_labels_and_subjects, stratified_group_split
from helper import LSTMModel, train_model, test_model
from torch.utils.data import TensorDataset, DataLoader
import torch

X = load_blockwise_sequences("adl_data_split_blocks.csv")
print(X.shape)

labels, subject_ids = load_labels_and_subjects("adl_labels.csv")
print(labels.shape)
print(subject_ids.shape)

print("Min label:", labels.min().item(), "Max label:", labels.max().item())

X_train, X_test, y_train, y_test = stratified_group_split(X, labels, subject_ids)

# Convert to TensorDatasets
train_dataset = TensorDataset(torch.stack([X[i] for i in X_train.indices]), y_train)
val_dataset = TensorDataset(torch.stack([X[i] for i in X_test.indices]), y_test)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, patience=10)
test_model(model, val_loader, device)