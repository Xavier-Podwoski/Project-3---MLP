import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# seeds
torch.manual_seed(16)
np.random.seed(16)


# ------------------------------------------------
# DATASET CLASS
# ------------------------------------------------
class MRIDataset(Dataset):
    def __init__(self, img_path, labels, t_size=100000):
        self.img_path = img_path
        self.labels = labels
        self.t_size = t_size

        self.images = []

        for path in img_path:
            img = self.preprocess_data(path)
            self.images.append(img)

        self.images = np.array(self.images)

        # normalize
        for i in range(len(self.images)):
            mean = self.images[i].mean()
            std = self.images[i].std() + 1e-8
            self.images[i] = (self.images[i] - mean) / std

    def preprocess_data(self, path):
        nii_img = nib.load(path)
        img_data = nii_img.get_fdata()
        img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)

        img_flat = img_data.flatten()

        # pad / crop
        if len(img_flat) > self.t_size:
            img_flat = img_flat[:self.t_size]
        elif len(img_flat) < self.t_size:
            img_flat = np.pad(img_flat, (0, self.t_size - len(img_flat)), mode='constant')

        return img_flat

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.images[idx]),
            torch.LongTensor([self.labels[idx]])
        )


# ------------------------------------------------
# MODEL
# ------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------
# UTIL FUNCTIONS
# ------------------------------------------------
def get_data_paths(data_dir):
    ad_dir = os.path.join(data_dir, "ADNI", "AD")
    cn_dir = os.path.join(data_dir, "ADNI", "CN")

    ad_paths = sorted([os.path.join(ad_dir, f) for f in os.listdir(ad_dir) if f.endswith('.npy') or f.endswith('.nii')])
    cn_paths = sorted([os.path.join(cn_dir, f) for f in os.listdir(cn_dir) if f.endswith('.npy') or f.endswith('.nii')])

    ad_labels = [1] * len(ad_paths)
    cn_labels = [0] * len(cn_paths)

    return ad_paths, cn_paths, ad_labels, cn_labels


def split_data(ad_paths, cn_paths, ad_labels, cn_labels, test_size=3):
    ad_t_path, ad_test_path, ad_t_label, ad_test_label = train_test_split(
        ad_paths, ad_labels, test_size=test_size, random_state=16, shuffle=True
    )

    cn_t_path, cn_test_path, cn_t_label, cn_test_label = train_test_split(
        cn_paths, cn_labels, test_size=test_size, random_state=16, shuffle=True
    )

    t_path = ad_t_path + cn_t_path
    t_label = ad_t_label + cn_t_label

    test_path = ad_test_path + cn_test_path
    test_label = ad_test_label + cn_test_label

    return t_path, test_path, t_label, test_label


# ------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------
def train_model(model, t_load, num_epochs=200, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        t_loss = 0.0
        t_correct = 0
        t_total = 0

        for images, labels in t_load:
            images, labels = images.to(device), labels.to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            t_total += labels.size(0)
            t_correct += (predicted == labels).sum().item()

        train_acc = 100 * t_correct / t_total

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss: {t_loss/len(t_load):.4f}, Acc: {train_acc:.2f}%")

    return model


# ------------------------------------------------
# EVALUATION
# ------------------------------------------------
def eval_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).squeeze()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {acc:.2f}% || Precision: {precision:.2f}% "
          f"|| Recall: {recall:.2f}% || F1: {f1:.2f}%")

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    return acc, precision, recall, f1, cm


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = "./"
    ad_paths, cn_paths, ad_labels, cn_labels = get_data_paths(data_dir)

    print(f"Data: {len(ad_paths)} AD, {len(cn_paths)} CN")

    t_path, test_path, t_label, test_label = split_data(
        ad_paths, cn_paths, ad_labels, cn_labels, test_size=3
    )

    print(f"Train: {len(t_path)} | Test: {len(test_path)}")

    train_dataset = MRIDataset(t_path, t_label)
    test_dataset = MRIDataset(test_path, test_label)

    input_size = train_dataset.images.shape[1]
    print(f"Size of Input: {input_size}")

    t_load = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = MLPClassifier(input_size).to(device)
    print("Training...")
    train_model(model, t_load, num_epochs=200, device=device)

    eval_model(model, test_loader, device=device)


if __name__ == "__main__":
    main()
