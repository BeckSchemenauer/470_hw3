from preprocessing import load_blockwise_sequences, load_labels_and_subjects, stratified_group_split, \
    oversample_minority_classes, print_class_distribution
from helper import LSTMModel, CNN1DModel, train_model, test_model
from torch.utils.data import TensorDataset, DataLoader
import torch
from itertools import product
import os
import sys
from datetime import datetime
from io import StringIO


def run_experiment(
        batch_size=64,
        learning_rate=0.001,
        hidden_layers=[250, 250, 50],
        dropout_rate=0.4,
        epochs=200,
        patience=20,
        model_type='lstm',
):
    # setup logging
    os.makedirs("f8_models", exist_ok=True)
    buffer = StringIO()
    sys.stdout = buffer  # redirect prints

    # load and split data
    X = load_blockwise_sequences("fall_data_split_blocks.csv")
    labels, subject_ids = load_labels_and_subjects("fall_labels.csv")
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
    if model_type == 'lstm':
        model = LSTMModel(hidden_layers=hidden_layers, dropout_rate=dropout_rate)
    elif model_type == 'cnn':
        model = CNN1DModel(hidden_layers=hidden_layers, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: {model_type.upper()}, Layers: {hidden_layers}, Dropout: {dropout_rate}, LR: {learning_rate}")

    # train and get final validation accuracy
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=epochs, patience=patience)
    final_acc = test_model(model, val_loader, device, return_acc=True)
    print(f"Final Validation Accuracy: {final_acc:.4f}")

    # generate filename
    hp_desc = f"{model_type}_bs{batch_size}_lr{learning_rate}_drop{dropout_rate}_layers{'-'.join(map(str, hidden_layers))}"
    filename = f"{final_acc:.4f}_{hp_desc}.txt"
    path = os.path.join("f8_models", filename)

    # write logs to file
    with open(path, "w") as f:
        f.write(buffer.getvalue())

    sys.stdout = sys.__stdout__  # reset stdout
    print(f"Saved output to {path}")


def run_hyperparameter_search():
    # define hyperparameter options
    batch_sizes = [64, 128]
    learning_rates = [.01, .001, .0005, 0.0001]
    hidden_layer_options = [[500], [500, 250], [250, 50], [250, 250, 50], [64, 32, 32], [64, 64], [128, 64, 64, 32]]
    dropout_rates = [0.3, 0.4, 0.5]
    model_types = ['lstm', 'cnn']
    epochs = 100
    patience = 10

    # create all combinations including model_type
    combinations = list(product(batch_sizes, learning_rates, hidden_layer_options, dropout_rates, model_types))

    for i, (batch_size, lr, hidden_layers, dropout, model_type) in enumerate(combinations):
        print(f"\nRunning configuration {i+1}/{len(combinations)}")
        print(f"Model: {model_type.upper()}, Batch Size: {batch_size}, LR: {lr}, Layers: {hidden_layers}, Dropout: {dropout}")

        run_experiment(
            batch_size=batch_size,
            learning_rate=lr,
            hidden_layers=hidden_layers,
            dropout_rate=dropout,
            epochs=epochs,
            patience=patience,
            model_type=model_type
        )



run_hyperparameter_search()
