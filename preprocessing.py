import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Subset
from collections import Counter
from imblearn.over_sampling import SMOTE


def create_file1(input_csv, output_csv):
    df = pd.read_csv(input_csv, header=0)
    split_rows = []

    for _, row in df.iterrows():
        subrows = [row[i * 151:(i + 1) * 151].values for i in range(3)]
        split_rows.extend(subrows)

    pd.DataFrame(split_rows).to_csv(output_csv, header=False, index=False)


def create_file2(input_csv, output_csv):
    df = pd.read_csv(input_csv, header=0)
    split_rows = []

    for _, row in df.iterrows():
        subrows = [row[i::3].values for i in range(3)]
        split_rows.extend(subrows)

    pd.DataFrame(split_rows).to_csv(output_csv, header=False, index=False)


def plot_first_three_rows(csv_file, title):
    df = pd.read_csv(csv_file, header=None)
    rows_to_plot = df.iloc[0:3]  # rows 1-3 (0-indexed)

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(rows_to_plot.iloc[i], label=f'Row {i + 1}')

    plt.title(f'Rows 1–3 from {title}')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# create_file1("fall_data.csv", "fall_data_split_blocks.csv")
# create_file2("adl_data.csv", "adl_data_split_stride.csv")
#
# plot_first_three_rows("adl_data_split_blocks.csv", "Block-wise (151x3 chunks)")
# plot_first_three_rows("adl_data_split_stride.csv", "Stride-wise (every 3rd element)")

def load_blockwise_sequences(csv_file):
    df = pd.read_csv(csv_file, header=None)
    sequences = []

    # group every 3 rows as one sample
    for i in range(0, len(df), 3):
        x = df.iloc[i].values  # row a
        y = df.iloc[i + 1].values  # row b
        z = df.iloc[i + 2].values  # row c

        # stack as (151, 3): [[x1, y1, z1], ..., [x151, y151, z151]]
        sequence = np.stack([x, y, z], axis=1)
        sequences.append(sequence)

    # return as torch tensor (N, 151, 3)
    return torch.tensor(np.array(sequences), dtype=torch.float32)


def load_labels_and_subjects(label_csv):
    df = pd.read_csv(label_csv, header=0)

    # convert labels to zero-indexed classes: 1–9 --> 0–8
    # or for fall dataset: 1–8 --> 0–7
    labels = df.iloc[:, 0].astype(int) - 1
    subjects = df.iloc[:, 1].values

    labels = torch.tensor(labels.values, dtype=torch.long)
    return labels, subjects


def stratified_group_split(X, labels, subject_ids, n_splits=5, fold_idx=0):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = list(sgkf.split(X, labels.numpy(), groups=subject_ids))

    train_idx, val_idx = indices[fold_idx]

    train_dataset = Subset(X, train_idx)
    val_dataset = Subset(X, val_idx)

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    return train_dataset, val_dataset, train_labels, val_labels


def oversample_minority_classes(X, y):
    # count how many samples there are per class
    class_counts = Counter(y.tolist())
    max_count = max(class_counts.values())

    # collect indices of each class
    indices_by_class = {cls: (y == cls).nonzero(as_tuple=True)[0] for cls in class_counts}

    # list to hold all oversampled indices
    all_indices = []

    # for each class, repeat its indices to match the max class count
    for cls, indices in indices_by_class.items():
        num_to_add = max_count - len(indices)

        # randomly sample with replacement to add up to max_count
        extra_indices = indices[torch.randint(len(indices), (num_to_add,))]
        oversampled = torch.cat([indices, extra_indices])
        all_indices.append(oversampled)

    # combine all class indices and shuffle
    final_indices = torch.cat(all_indices)
    shuffled_indices = final_indices[torch.randperm(len(final_indices))]

    X_balanced = X[shuffled_indices]
    y_balanced = y[shuffled_indices]

    return X_balanced, y_balanced


def print_class_distribution(label_tensor, name):
    unique, counts = np.unique(label_tensor.numpy(), return_counts=True)
    total = len(label_tensor)
    print(f"\n{name} class distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({(c / total * 100):.2f}%)")


def normalize_axes_separately(X_train, X_test):

    X_train_norm = X_train.clone()
    X_test_norm = X_test.clone()

    for axis in range(3):
        # (N, 151, 3) → (N*151,) for axis
        axis_values = X_train[:, :, axis].reshape(-1)
        mean = axis_values.mean()
        std = axis_values.std()

        X_train_norm[:, :, axis] = (X_train[:, :, axis] - mean) / (std + 1e-8)
        X_test_norm[:, :, axis] = (X_test[:, :, axis] - mean) / (std + 1e-8)

    return X_train_norm, X_test_norm

def smote_expand_dataset(X_tensor, y_tensor, target_per_class=5000):

    X_flat = X_tensor.numpy().reshape(X_tensor.size(0), -1)  # (N, 453)
    y_np = y_tensor.numpy()

    # Define how many samples each class should have
    class_counts = Counter(y_np)
    smote_target = {cls: target_per_class for cls in class_counts}

    sm = SMOTE(sampling_strategy=smote_target, random_state=42, n_jobs=-1)
    X_resampled, y_resampled = sm.fit_resample(X_flat, y_np)

    # Reshape back to (N', 151, 3)
    X_res_tensor = torch.tensor(X_resampled.reshape(-1, 151, 3), dtype=torch.float32)
    y_res_tensor = torch.tensor(y_resampled, dtype=torch.long)

    return X_res_tensor, y_res_tensor