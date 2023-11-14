import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


metadata_file = 'FSDKaggle2018.meta/train_post_competition.csv'
model_path = 'models/modelV1.2.pth'

# Load labels
train_metadata = pd.read_csv(metadata_file)
train_metadata = train_metadata[['fname', 'freesound_id', 'label']]
labels = np.unique(train_metadata.label.values)

# Load the model
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Similar class found in model's SoundDataset class (edited to acommodate 1 file instead of DataFrame)
class SoundFile(File):
    def __init__(self, path, test=False):
        super(SoundFile, self).__init__()
        self.path = path
        self.test = test

    def __getitem__(self, idx):
        signal, _ = librosa.load(path)
        signal = librosa.feature.melspectrogram(y=signal)
        signal = librosa.power_to_db(signal, ref=np.max)

        try:
            resized = cv2.resize(signal, (IMG_SIZE[1], IMG_SIZE[0]))
        except Exception as e:
            print(path)
            print(str(e))
            resized = np.zeros(shape=(IMG_SIZE[1], IMG_SIZE[0]))

        X = np.stack([resized] * 3)
        X = torch.tensor(X, dtype=torch.float32)

        if not self.test:
            y = label_encoder[label]
            return X, y
        else:
            return X


def predict(file_path):

    test_data = SoundFile(file_path, test=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    predictions = torch.tensor([])
    
    model.eval()
    y_hat = model(test_data)
    predictions = torch.cat([predictions, y_hat.cpu()])

    predictions = F.softmax(predictions, dim=1).detach().numpy()
    
    with torch.no_grad():
        for X in test_loader:
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
    
