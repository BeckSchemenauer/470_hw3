from preprocessing import load_blockwise_sequences, load_labels_and_subjects, stratified_group_split, oversample_minority_classes
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

# convert X_train indices into tensors
X_train_tensor = torch.stack([X[i] for i in X_train.indices])
y_train_tensor = y_train

X_balanced, y_balanced = oversample_minority_classes(X_train_tensor, y_train_tensor)

# create balanced train loader
train_dataset = TensorDataset(X_balanced, y_balanced)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# validation loader stays the same (no balancing)
val_dataset = TensorDataset(torch.stack([X[i] for i in X_test.indices]), y_test)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = LSTMModel(hidden_layers=[250, 250, 50], dropout_rate=0.4)

# Choose optimizer
use_sgd = False  # set True to use SGD

if use_sgd:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.1)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1)

criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=200, patience=20)
test_model(model, val_loader, device)
