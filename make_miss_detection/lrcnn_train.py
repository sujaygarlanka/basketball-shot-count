import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torch.utils import data

class BallDataset(data.Dataset):
    def __init__(self, data_path, label_path, video_path, transform=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.video_paths = np.load(video_path)
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, self.labels[index], self.video_paths[index]

    def __len__(self):
        return len(self.labels)

class LCRNN(nn.Module):
    # input_dim: Number of features for each frames
    # hidden_dim: LSTM hidden units
    # num_layers: Number of lstm units stacked on top
    # s_len: sequence length for input i.e. number of frames in a video
    def __init__(self, input_dim=2048, hidden_dim=128, num_layers=1, seq_len=16, dropout_rate=0.3, batch_size = 1, num_classes = 2):
        super(LCRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        hidden_state = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        cell_state = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)
        # N x 16 x 2048
        x = self.dropout(x)
        # N x 16 x 2048
        x, hidden = self.lstm(x, hidden)
        # N x 16 x 128
        x = self.dropout(x)
        # N x 16 x 128
        x = self.fc1(x)
        # N x 16 x 2
        x = self.softmax(x)

        # # N x 16 x 2
        # x = torch.mean(x, 1)

        # Get last layer
        # N x 16 x 2
        x = x[:, -1, :]
        # N x 1 x 2
        x = torch.squeeze(x, 1)
        # N x 2
        return x, hidden


class Trainer():
    def __init__(self):
        data = BallDataset('./processed_data/data.npy', './processed_data/labels.npy', './processed_data/video_paths.npy')
        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
        self.test_dataset = BallDataset('./processed_test_data/data.npy', './processed_test_data/labels.npy', './processed_test_data/video_paths.npy')

    def train(self, NET_PATH, num_epochs, batch_size):
        net = LCRNN(batch_size=batch_size)
        train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, video_paths = data

                # zero the parameter gradients
                net.zero_grad()
                # forward + backward + optimize
                outputs, hidden = net(inputs.float())
                _, class_labels = torch.max(labels.data, 1)
                loss = criterion(outputs, class_labels)
                loss.backward()
                optimizer.step()
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))

        torch.save(net.state_dict(), NET_PATH)

    def evaluate(self, NET_PATH):
        batch_size = 1
        net = LCRNN(batch_size=batch_size)
        net.load_state_dict(torch.load(NET_PATH))
        loader = DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

        num_classes = 2
        classes = ('make', 'miss')
        correct = 0
        total = 0
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))
        class_correct_path = [[] for i in range(num_classes)]
        class_wrong_path = [[] for i in range(num_classes)]
        # class_wrong_label = [[] for i in range(num_classes)]
        with torch.no_grad():
            for data in loader:
                images, labels, video_paths = data
                outputs, hidden = net(images.float())
                _, formatted_labels = torch.max(labels.data, 1)
                _, predicted = torch.max(outputs, 1)
                for i in range(batch_size):
                    label = formatted_labels[i]
                    if predicted[i] == label:
                        correct += 1
                        class_correct[label] += 1
                        class_correct_path[label].append(video_paths[i].item())
                    else:
                        class_wrong_path[label].append(video_paths[i].item())
                        # class_wrong_label[label].append(classes[predicted[i].item()])
                    class_total[label] += 1
                total += formatted_labels.size(0)
        
        print('Accuracy of the network on the %d images: %d %%' % (total, 100 * correct / total))
        for i in range(num_classes):
            print('Accuracy of %d %5s : %2d %%' % (class_total[i], classes[i], 100 * class_correct[i] / class_total[i]))
        # np.save('paths_correct.npy', class_correct_path)
        # np.save('paths_wrong.npy', class_wrong_path)
        # np.save('labels_wrong.npy', class_wrong_label)

    def display_video(self, path):
        vidcap = cv2.VideoCapture(path)
        frames = []
        # extract frames
        while True:
            success, image = vidcap.read()
            if not success:
                break
            frames.append(image)
        # downsample if desired and necessary to num_frames
        num_frames = 16
        if num_frames < len(frames):
            skip = len(frames) // num_frames
            frames = [frames[i] for i in range(0, len(frames), skip)]
            frames = frames[:num_frames]
        for frame in frames:
            cv2.imshow('image', frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        return frames

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train('./trained_net_paths/trained_net_last_layer_unclean_data.pth', num_epochs=10, batch_size=60)
    trainer.evaluate('./trained_net_paths/trained_net_last_layer_unclean_data.pth')

    # paths_correct = np.load('paths_correct.npy', allow_pickle=True)
    # paths_wrong = np.load('paths_wrong.npy', allow_pickle=True)
    # i = 3
    # print(paths_wrong[1][i])
    # trainer.display_video(paths_wrong[1][i])
