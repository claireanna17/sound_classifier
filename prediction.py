import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import pandas as pd
import numpy as np
import librosa
import cv2

IMG_SIZE = (128, 128)
batch_size = 1

metadata_file = 'train_post_competition.csv'
model_path = 'modelV1.2.pth'

# Load labels
train_metadata = pd.read_csv(metadata_file)
train_metadata = train_metadata[['fname', 'freesound_id', 'label']]
labels = np.unique(train_metadata.label.values)

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

# Initialize model
WeightsEnum.get_state_dict = get_state_dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.classifier[1] = torch.nn.Linear(1280, 41)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
cost = torch.nn.CrossEntropyLoss()

# Load saved weights
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Similar class found in model's SoundDataset class (edited to acommodate 1 file instead of DataFrame)
class SoundDataset(Dataset):
    def __init__(self, dataframe, path, test=False):
        super(SoundDataset, self).__init__()
        self.size = 1
        self.path = path
        self.test = test

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        signal, _ = librosa.load(self.path)
        signal = librosa.feature.melspectrogram(y=signal)
        signal = librosa.power_to_db(signal, ref=np.max)

        try:
            resized = cv2.resize(signal, (IMG_SIZE[1], IMG_SIZE[0]))
        except Exception as e:
            print(self.path)
            print(str(e))
            resized = np.zeros(shape=(IMG_SIZE[1], IMG_SIZE[0]))

        X = np.stack([resized] * 3)
        X = torch.tensor(X, dtype=torch.float32)

        return X


def predict(file_path):

    test_dataset = SoundDataset(1, file_path, test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = torch.tensor([])
    
    model.eval()
    with torch.no_grad():
        for X in test_loader:
          X = X.to(device)
          y_hat = model(X)
          predictions = torch.cat([predictions, y_hat.cpu()])
    
    predictions = F.softmax(predictions, dim=1).detach().numpy()   
    
    p = predictions[0, :]
    idx = np.argmax(p)
    result = labels[idx]
    
    return result


if __name__ == "__main__":
    file = " "
    predict(file)
    
