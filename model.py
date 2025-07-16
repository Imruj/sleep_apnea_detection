import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

# ----- Dataset -----
class SleepApneaAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.classes = {"normal": 0, "apnea": 1}
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        for label_name in self.classes:
            class_path = os.path.join(root_dir, label_name)
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    self.data.append(os.path.join(class_path, file))
                    self.labels.append(self.classes[label_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(file_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        mel = self.transform(waveform)
        mel_db = self.db_transform(mel)
        mel_db = mel_db.mean(dim=0, keepdim=True)  # convert to mono

        return mel_db, label

# ----- CNN Model -----
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B,16,32,64)
        x = self.pool(F.relu(self.conv2(x)))  # (B,32,16,32)
        x = self.pool(F.relu(self.conv3(x)))  # (B,64,8,16)
        x = self.adapt_pool(x)               # (B,64,1,1)
        x = x.view(x.size(0), -1)            # (B,64)
        return self.fc(x)

# ----- Evaluate -----
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), torch.tensor(y).to(device)
            out = model(X)
            pred = torch.argmax(out, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# ----- Predict any file -----
def predict_file(file_path, model, device):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
    mel = transform(waveform)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    mel = mel.mean(dim=0, keepdim=True).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(mel)
        prob = torch.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        classes = ["normal", "apnea"]
        print(f"Predict: {classes[pred_class]} (confidence: {prob[0][pred_class]:.2f})")

# ----- Main Training -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    train_dataset = SleepApneaAudioDataset("data/train")
    val_dataset = SleepApneaAudioDataset("data/val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), torch.tensor(y).to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "apnea_model.pth")
    print("Model saved as apnea_model.pth")

    # Confusion Matrix
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            output = model(X)
            pred = torch.argmax(output, dim=1).cpu().numpy()
            y_true.extend(y)
            y_pred.extend(pred)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Predict sample file (optional)
    # predict_file("your_test_file.wav", model, device)

if __name__ == "__main__":
    main()
