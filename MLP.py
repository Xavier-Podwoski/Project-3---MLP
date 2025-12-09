import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, 
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(16)
np.random_seed(16)

class MRIData(Dataset):
    def __init__(self, img_path, labels, t_size=10000):
      self.img_path = img_patt
      self.labels = labels
      self.t_size = t_size
    
      self.images = []
      self.labels = labels
      self.t_size = t_size

      for path in img_path:
          img = self.preprocess_data(path)
          self.images.append(img)

          self.images = np.array(self.images)

        # normalize // reduce variance
          for i in range(len(self.images)):
            mean = self.images[i].mean()
            std = self.images[i].std() + 1e-8
            self.images[i] = (self.images[i] - mean) / std

    def preprocess_data(self, path):
        nii_img = nib.load(path)
        img_data = nii_img.get_fdata()
        img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)
      
        # fixed size
        img_flat = img_data.flatten()

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
      
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
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
      
    def get_data_paths(data_dir):
        ad_dir = os.path.join(data_dir, 'ADNI', 'AD')
        cn_dir = os.path.join(data_dir, 'ADNI', 'CN')

        ad_paths = sorted([os.path.join(ad_dir, f) for f in os.listdir(ad_dir) if f.endswith('.npy')])
        cn_paths = sorted([os.path.join(cn_dir, f) for f in os.listdir(cn_dir) if f.endswith('.npy')])

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

        t_path  = ad_t_path  + cn_t_path
        t_label = ad_t_label + cn_t_label

        test_path  = ad_test_path  + cn_test_path
        test_label = ad_test_label + cn_test_label

        return t_path, test_path, t_label, test_label
    
    def train_model(model, t_load, num_epochs=200, device='cuda'):
      crit = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

      for epoch in range(num_epochs):
          t_loss = 0.0
          t_correct = 0
          t_total = 0

      for images, labels in t_load:
          images, labels = images.to(device), labels.to(device)

          optimizer.zero_grad()
          outputs = model(images)
          loss = crit(outputs, labels)
          loss.backward()
          optimizer.step()

          t_loss += loss.item()
          _, predicted = torch.max(outputs.data, 1)
          t_total += labels.size(0)
          t_correct += (predicted == labels).sum().item()
          
          train_acc = 100 * t_correct / t_total

          if (epoch + 1) % 20 == 0:
              print(f"Epoch {epoch+1}/{num_epochs}: "
                    f"Loss: {t_loss/len(t_load):.4f}, "
                    f"Acc: {train_acc:.2f}%")

        return model

    def eval_model(model, test_loader, device='cuda'):
        model = model.to(device)
        model.eval()
        all_preds = []
        all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
          
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    accuracy = 100 * correct / total

    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {accuracy:.2f}% || Precision: {precision:.2f}% || Recall: {recall:.2f}% || F1: {f1:.2f}%")

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1, cm

    def main():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        ad_paths, cn_paths, ad_labels, cn_labels = get_data_paths()
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

        accuracy, precision, recall, f1, cm = eval_model(model, test_loader, device=device)

        #model saving
        '''
        torch.save(model.state_dict(), 'mlp_model.pth'
        print('Model is successfully saved.')
        '''

    if __name__ == "__main__":
        main()
