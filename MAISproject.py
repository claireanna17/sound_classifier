import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import pandas as pd
import numpy as np
import librosa
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
path='C:/Users/ASUS/Downloads/FSDKaggle2018/'

# Read metadata file
metadata_file = path + 'FSDKaggle2018.meta/train_post_competition.csv'
# print(metadata_file)
# contents = os.listdir('/content/drive/MyDrive/FSDKaggle2018')

# for item in contents:
#     print(item)
train = pd.read_csv(metadata_file)

# Take relevant columns (features, labels)
train = train[['fname', 'freesound_id', 'label']]
print(train)

# Audio training data path
data_path = path + 'FSDKaggle2018l.audio_train/'
IMG_SIZE = (128, 128)
TRAIN_PATH = path + '/FSDKaggle2018.audio_train/'
TEST_PATH = path + '/FSDKaggle2018.audio_test/'
batch_size = 32
class SoundDataset(Dataset):
    def __init__(self, dataframe, path, test=False):
        super(SoundDataset, self).__init__()
        self.dataframe = dataframe
        self.path = path
        self.test = test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        file_path = self.dataframe.fname.values[idx]
        label = self.dataframe.label.values[idx]
        path = (TEST_PATH if self.test else TRAIN_PATH) + file_path
        signal, _ = librosa.load(path)
        signal = librosa.feature.melspectrogram(y=signal)
        signal = librosa.power_to_db(signal, ref=np.max)

        try:
            resized = cv2.resize(signal, (IMG_SIZE[1], IMG_SIZE[0]))
        except Exception as e:
            print(path)
            print(str(e))
            resized = np.zeros(shape=(IMG_SIZE[1], IMG_SIZE[0]))

        X = np.stack([resized] * 3)  # Дублирование каналов
        X = torch.tensor(X, dtype=torch.float32)

        if not self.test:
            y = label_encoder[label]
            return X, y
        else:
            return X

x_train, x_validation, y_train, y_validation = train_test_split(train, train, test_size=0.2, shuffle=True, random_state=5)
train_dataset = SoundDataset(x_train, TRAIN_PATH)
val_dataset = SoundDataset(x_validation, TRAIN_PATH)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(x_train.head())
print(y_train.head())
labels = np.unique(train.label.values)
label_encoder = {label:i for i, label in enumerate(labels)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torchvision.models.efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 41)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(optimizer)
cost = torch.nn.CrossEntropyLoss()
print(cost)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torchvision.models.efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 41)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(optimizer)
cost = torch.nn.CrossEntropyLoss()
print(cost)
def train(epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = cost(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            train_correct += torch.sum(pred.argmax(1) == y).item()

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = cost(pred, y)
                val_loss += loss.item() * X.size(0)
                val_correct += torch.sum(pred.argmax(1) == y).item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        print("Epoch {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))

num_epochs = 15
train(num_epochs)
