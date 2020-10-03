import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torch.utils import data
from torchvision import transforms
from data_utils import VideoProcessing
from lrcnn_train import LCRNN, BallDataset
from PIL import Image

class ActionDetection():
    def __init__(self, lcrnn_path, classes, batch_size=1):
        self.classes = classes
        Inception_V3 = models.inception_v3(pretrained=True)
        Inception_V3.fc = nn.Identity()
        Inception_V3.eval()
        self.Inception_V3 = Inception_V3

        net = LCRNN(batch_size=batch_size)
        net.load_state_dict(torch.load(lcrnn_path))
        self.LCRNN = net
        
    # Takes in RGB frames and pre-processes them for the CNN, which is currently Inception V3
    def pre_process_CNN(self, frames):
        sequence = []
        for img in frames:
            PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                        0.229, 0.224, 0.225])                       
            ])
            input_tensor = preprocess(PIL_image)
            # print(input_tensor.numpy().shape)
            # cv2.imshow('image', input_tensor.numpy())
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            sequence.append(input_tensor)
        return torch.stack(sequence)

    def convert_tensor_to_classes(self, x):
        predicted = torch.max(x, 1)
        return predicted

    def classify(self, x):
        # move the input and model to GPU for speed if available
        x = self.pre_process_CNN(x)
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.Inception_V3.to('cuda')
        with torch.no_grad():
            x = self.Inception_V3(x)
        # create a mini-batch of 1 as expected by the model
        x = x.unsqueeze(0)
        x, hidden = self.LCRNN(x)
        return x

if __name__ == "__main__":
    new_frames = VideoProcessing().get_frames_from_video('/Users/SujayGarlanka/Desktop/IMG_0568_original.avi')
    # np.save('frames.npy', np.array(frames))
    frames = np.load('frames.npy')
    # for frame in frames:
    #     PIL_image = Image.fromarray(np.uint8(frame)).convert('RGB')
    #     preprocess = transforms.Compose([
    #         transforms.Resize(299),
    #         transforms.CenterCrop(299),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
    #         #                         0.229, 0.224, 0.225]),
    #         transforms.ToPILImage()
    #     ])
    #     # print(frame.shape)
    #     # pil = preprocess(PIL_image)
    #     # pil.show()
    #     # break
    #     cv2.imshow('image', frame)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    action_detection = ActionDetection('./trained_net_paths/trained_net.pth', ['make', 'miss'])
    out = action_detection.classify(frames)
    print(out)
