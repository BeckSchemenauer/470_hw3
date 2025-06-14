from preprocessing import load_blockwise_sequences, load_labels_and_subjects, stratified_group_split, \
    oversample_minority_classes, print_class_distribution, smote_expand_dataset, normalize_axes_separately
from helper import LSTMModel, CNN1DModel, TransformerModel, train_model, test_model
from torch.utils.data import TensorDataset, DataLoader
import torch
from itertools import product


def run_experiment(
        batch_size=64,
        learning_rate=0.001,
        hidden_layers=[250, 250, 50],
        conv_config=None,
        dropout_rate=0.4,
        epochs=200,
        patience=20,
        model_type='lstm',
):
    # load and split data
    X = load_blockwise_sequences("fall_data_split_blocks.csv")
    labels, subject_ids = load_labels_and_subjects("fall_labels.csv")
    X_train, X_test, y_train, y_test = stratified_group_split(X, labels, subject_ids)

    #print(f"Loaded data: {X.shape}")
    #print(f"Min label: {labels.min().item()}, Max label: {labels.max().item()}")

    # oversample training data
    X_train_tensor = torch.stack([X[i] for i in X_train.indices])
    y_train_tensor = y_train
    X_balanced, y_balanced = oversample_minority_classes(X_train_tensor, y_train_tensor)

    # sanity check: shuffle labels to test for data leakage or broken signal
    if model_type.endswith("_shuffled"):
        model_type = model_type.split('_')[0]
        print("Shuffling training labels for sanity check")
        idx = torch.randperm(len(y_balanced))
        y_balanced = y_balanced[idx]

    # extract raw test tensor
    # X_test_tensor = torch.stack([X[i] for i in X_test.indices])

    # expand dataset with SMOTE to ~5000 samples per class
    # X_balanced, y_balanced = smote_expand_dataset(X_balanced, y_balanced, target_per_class=5000)

    # normalize each axis using train stats
    # X_balanced, X_test_tensor = normalize_axes_separately(X_balanced, X_test_tensor)

    #print_class_distribution(y_balanced, "Train")
    #print_class_distribution(y_test, "Test")

    # train and validation datasets/loaders
    train_dataset = TensorDataset(X_balanced, y_balanced)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.stack([X[i] for i in X_test.indices]), y_test)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model and training setup
    if model_type == 'lstm':
        model = LSTMModel(hidden_layers=hidden_layers, dropout_rate=dropout_rate, out_features=8)
    elif model_type == 'cnn':
        model = CNN1DModel(
            conv_config=conv_config,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            input_length=151,
            input_channels=3,
            out_features=8,
            pool_every=3
        )
    elif model_type == 'transformer':
        model = TransformerModel(hidden_layers=hidden_layers, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: {model_type.upper()}, Layers: {hidden_layers}, Dropout: {dropout_rate}, LR: {learning_rate}")

    # train and get final validation accuracy
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=epochs, patience=patience)
    final_acc = test_model(model, val_loader, device, print_output=True)
    print(f"Final Validation Accuracy: {final_acc:.4f}")


def run_hyperparameter_search():
    # define hyperparameter options
    batch_sizes = [64, 100]
    learning_rates = [.0002,]
    hidden_layer_options = [[64], [64], [64]]
    dropout_rates = [.85,]
    model_types = ['cnn']
    epochs = 200
    patience = 15

    conv_config_options = [
        #[(64, 12), (128, 12), (256, 12), (256, 12)],
        #[(64, 10), (128, 10), (256, 10), (512, 10)],
        #[(128, 10), (256, 10), (512, 10), (1024, 10)],
        #[(64, 10), (128, 10), (256, 10), (512, 10), (512, 10)],
        #[(12, 10), (128, 10), (256, 10), (512, 10), (1024, 10)], # best so far (w/ 0.7 dropout)
        #[(12, 10), (128, 10), (256, 10), (512, 10), (1024, 10), (2048, 10)], # best so far (w/ 0.8 dropout)
        # [(8, 10), (32, 10), (64, 10), (128, 10), (512, 10), (1024, 10)], # best so far (w/ 0.8 dropout) !!!!
        [(8, 10), (32, 10), (128, 10), (512, 10), (128, 10)],  # (actually best so far 0.55)
        #[(8, 10), (32, 10), (128, 10), (32, 10)],
        #[(256, 10), (512, 10), (128, 10), (32, 10)],
        #[(8, 10), (256, 10), (512, 10), (128, 10)],
        #[(8, 10), (64, 10), (256, 10), (64, 10), (128, 10)],
        #[(512, 10), (128, 10), (64, 10), (32, 10)],
        #[(12, 10), (128, 10), (256, 10), (512, 10), (2048, 10)],
        # [(64, 8), (128, 8), (256, 8), (256, 8)],
        # [(64, 6), (128, 6), (256, 6), (256, 6)],
        # [(64, 4), (128, 4), (256, 4), (256, 4)],
    ]

    # create all combinations including model_type
    combinations = list(product(
        batch_sizes,
        learning_rates,
        hidden_layer_options,
        conv_config_options,
        dropout_rates,
        model_types
    ))

    for i, (batch_size, lr, hidden_layers, conv_config, dropout, model_type) in enumerate(combinations):
        print(f"\nRunning configuration {i + 1}/{len(combinations)}")
        print(
            f"Model: {model_type.upper()}, Batch Size: {batch_size}, LR: {lr}, Conv: {conv_config}, Layers: {hidden_layers}, Dropout: {dropout}")

        run_experiment(
            batch_size=batch_size,
            learning_rate=lr,
            hidden_layers=hidden_layers,
            conv_config=conv_config,
            dropout_rate=dropout,
            epochs=epochs,
            patience=patience,
            model_type=model_type
        )


run_hyperparameter_search()
