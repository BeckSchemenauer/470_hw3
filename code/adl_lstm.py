from preprocessing import load_blockwise_sequences, load_labels_and_subjects, stratified_group_split, \
    oversample_minority_classes, print_class_distribution
from helper import LSTMModel, CNN1DModel, train_model, test_model
from torch.utils.data import TensorDataset, DataLoader
import torch
from itertools import product


def run_experiment(
        batch_size=64,
        learning_rate=0.001,
        hidden_layers=[250, 250, 50],
        dropout_rate=0.4,
        epochs=200,
        patience=20
):
    # load and split data
    X = load_blockwise_sequences("../data/adl_data_split_blocks.csv")
    labels, subject_ids = load_labels_and_subjects("adl_labels.csv")
    X_train, X_test, y_train, y_test = stratified_group_split(X, labels, subject_ids)

    print(f"Loaded data: {X.shape}")
    print(f"Min label: {labels.min().item()}, Max label: {labels.max().item()}")

    # oversample training data
    X_train_tensor = torch.stack([X[i] for i in X_train.indices])
    y_train_tensor = y_train
    X_balanced, y_balanced = oversample_minority_classes(X_train_tensor, y_train_tensor)

    print_class_distribution(y_balanced, "Train")
    print_class_distribution(y_test, "Test")

    # train and validation datasets/loaders
    train_dataset = TensorDataset(X_balanced, y_balanced)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.stack([X[i] for i in X_test.indices]), y_test)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model and training setup
    model = LSTMModel(hidden_layers=hidden_layers, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.1)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model architecture: {hidden_layers}, Dropout: {dropout_rate}, LR: {learning_rate}")

    # train and evaluate
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=epochs, patience=patience)
    test_model(model, val_loader, device)


def run_hyperparameter_search():
    # define hyperparameter options
    batch_sizes = [64]
    learning_rates = [0.0001]
    hidden_layer_options = [[250, 250, 50],]
    dropout_rates = [0.5]
    epochs = 100
    patience = 10

    # create all combinations
    combinations = list(product(batch_sizes, learning_rates, hidden_layer_options, dropout_rates))

    for i, (batch_size, lr, hidden_layers, dropout) in enumerate(combinations):
        print(f"\nRunning configuration {i+1}/{len(combinations)}")
        print(f"Batch Size: {batch_size}, Learning Rate: {lr}, Layers: {hidden_layers}, Dropout: {dropout}")

        run_experiment(
            batch_size=batch_size,
            learning_rate=lr,
            hidden_layers=hidden_layers,
            dropout_rate=dropout,
            epochs=epochs,
            patience=patience
        )


run_hyperparameter_search()
