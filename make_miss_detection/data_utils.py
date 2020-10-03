import os
from os import walk
import glob
import csv
import cv2
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import ffmpeg    


class VideoProcessing():
    def __init__(self):
        pass

    def get_frames_from_video(self, video_path, num_frames=16):
        vidcap = cv2.VideoCapture(video_path)
        frames = []
        # check if video requires rotation
        rotateCode = self.check_rotation(video_path)
        # extract frames
        while True:
            success, image = vidcap.read()
            if not success:
                break
            img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if rotateCode is not None:
                img_RGB = cv2.rotate(img_RGB, rotateCode) 
            frames.append(img_RGB)
        # downsample if desired and necessary
        if num_frames < len(frames):
            skip = len(frames) // num_frames
            frames = [frames[i] for i in range(0, len(frames), skip)]
            frames = frames[:num_frames]
        return frames

    def create_video_from_frames(self, path, frames, width, height, num_frames=16):
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), num_frames, (width, height))
        for i in range(len(frames)):
            # writing to a image array
            out.write(frames[i])
        out.release()

    def check_rotation(self, path_video_file):
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)

        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        rotateCode = None
        try:
            if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
            elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                rotateCode = cv2.ROTATE_180
            elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
        except KeyError:
            pass
        return rotateCode

    def show_video(self, frames):
        for frame in frames:
            cv2.imshow('image', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def data_augmentation():
    video = VideoProcessing()
    original_data_dir = 'original_test_data'
    processed_data_dir = 'processed_test_data'
    count = 1
    num_of_files = sum([len(files) for r, d, files in os.walk(original_data_dir)])
    dirs = ['make', 'miss']
    for dir in dirs:
        for file in glob.glob(f'./{original_data_dir}/{dir}/*.MOV'):
            frames = video.get_frames_from_video(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            resize = 299
            
            # Flip
            # Grayscale
            # Flip and grayscale
            original = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize)
            ])
            flip = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize)
            ])
            grayscale = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize)
            ])
            flip_grayscale = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize)
            ])
            # jitter = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05).get_params()
            transforms_dict = {'original': original, 'flip': flip, 'grayscale': grayscale, 'flip_grayscale': flip_grayscale}
            
            for key in transforms_dict:
                transformed_frames = []
                transform = transforms_dict[key]
                for frame in frames:
                    PIL_image = Image.fromarray(frame)
                    trans_frame = transform(PIL_image)
                    # Go from RGB to BGR to save video
                    transformed_frames.append(np.asarray(trans_frame)[:, :, ::-1])
                transformed_video_file_path = os.path.join(processed_data_dir, dir, filename + '_' + key + '.avi')
                video.create_video_from_frames(transformed_video_file_path, transformed_frames, resize, resize)
            print('Progress percent: ' + str(count/num_of_files * 100))
            count+=1

            
def convert_video_to_net_data():
    """
    This file imports RGB videos, breaks each one into its frames,
    and runs each frame through the Inception V3 net to get a features list
    for each frame. It finally concatenates the features list for all the frames in a video
    and adds it to a list of training, validation, or testing examples. Finally, the examples are saved to
    numpy files.
    """
    video = VideoProcessing()
    processed_data_dir = 'processed_test_data'
    Inception_V3 = models.inception_v3(pretrained=True)
    Inception_V3.fc = nn.Identity()
    Inception_V3.eval()
    num_frames = 16
    # First index is features of the video, second is the label, third is file path to the original video
    data = [[],[],[]]
    # Not perfectly precise because this also includes .DS_Store and other hidden files. Count will be a few off
    count = 1
    num_of_files = sum([len(files) for r, d, files in os.walk(processed_data_dir)])
    dirs = ['make', 'miss']
    for dir in dirs:
        for file in glob.glob(f'./{processed_data_dir}/{dir}/*.avi'):
            frames = video.get_frames_from_video(file)

            sequence = []
            for img in frames:
                PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                            0.229, 0.224, 0.225])
                ])
                input_tensor = preprocess(PIL_image)
                input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
                # move the input and model to GPU for speed if available
                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')
                    Inception_V3.to('cuda')
                with torch.no_grad():
                    features = Inception_V3(input_batch)
                sequence.append(features[0].tolist())  # only take first dimension

            data[0].append(sequence)
            data[1].append(get_one_hot_encoding(dir))
            data[2].append(file)
            print('Progress percent: ' + str(count/num_of_files * 100))
            count+=1

    np.save(os.path.join(processed_data_dir, 'data.npy'), np.array(data[0]))
    np.save(os.path.join(processed_data_dir, 'labels.npy'), np.array(data[1]))
    np.save(os.path.join(processed_data_dir, 'video_paths.npy'), np.array(data[2]))

# Encoding
# make = [1.0, 0.0]
# miss = [0.0, 1.0]
def get_one_hot_encoding(label):
    encoding = [0.0, 0.0]
    if label == 'make':
        encoding[0] = 1.0
    elif label == 'miss':
        encoding[1] = 1.0
    return encoding

if __name__ == "__main__":
    # data_augmentation()
    convert_video_to_net_data()