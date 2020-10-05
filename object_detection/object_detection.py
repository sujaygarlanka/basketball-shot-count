import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import TensorDataset, DataLoader
from coco_detection_dataset import CocoDetectionBoundingBox
from image_dataset import ImageFolder
import numpy as np
from model import YoloNetV3
from utils import post_process, untransform_bboxes, cxcywh_to_xywh, load_weights_from_original
from video_processing import convert_videos_to_images
from PIL import Image, ImageDraw, ImageFont


class ObjectDetection():

    def __init__(self, conf_thresh=0.8, nms_thresh=0.4, weights_path='./weights/yolov3_original.pth'):
        self.conf_thresh = conf_thresh
        self.nms_thres = nms_thresh
        self.net = YoloNetV3()
        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()
        self.classes = self.load_classes('./classes.txt')

    def detect(self, images, scales, paddings):
        with torch.no_grad():
            detections = self.net(images)
        detections = post_process(detections, True, self.conf_thresh, self.nms_thres)
        for detection, scale, padding in zip(detections, scales, paddings):
            detection[..., :4] = untransform_bboxes(detection[..., :4], scale, padding)
            cxcywh_to_xywh(detection)
        return detections

    def draw_result(self, img, boxes, show=False):
        if isinstance(img, torch.Tensor):
            transform = ToPILImage()
            img = transform(img)
        draw = ImageDraw.ImageDraw(img)
        for box in boxes:
            x, y, w, h = box[:4]
            x2 = x + w
            y2 = y + h
            draw.rectangle([x, y, x2, y2], outline='white', width=3)
            # Label box with class
            class_name = self.convert_one_hot_to_class(box[5:])
            font_size = 20
            class_font = ImageFont.truetype("./fonts/Roboto-Regular.ttf", font_size)
            label = class_name + " " + str(round(box[4].item(), 2))
            text_size = draw.textsize(label, font=class_font)
            draw.rectangle([x, y-text_size[1], x + text_size[0], y], fill='white')
            draw.text([x, y-font_size], label, font=class_font, fill='black')
        if show:
            img.show()
        return img
    
    def load_classes(self, path):
        """
        Loads class labels at 'path'
        """
        fp = open(path, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def convert_one_hot_to_class(self, label_tensor):
        value, index = torch.max(label_tensor, 0)
        return self.classes[index]
        

if __name__ == "__main__":
    # convert_videos_to_images('/Users/SujayGarlanka/Downloads/Originals/Make','/Users/SujayGarlanka/Projects/ML/basketball_shot_count/object_detection/images_to_label')
    net = ObjectDetection(conf_thresh=0.4, nms_thresh=0.4)
    
    # custom_dataset = CocoDetectionBoundingBox('./coco/val2017', './coco/instances_val2017.json', img_size=416)
    # custom_dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)
    # for i, data in enumerate(custom_dataloader):
    #     tensor, label = data
    #     print(label)
    #     break

    image_dataset = ImageFolder('/Users/SujayGarlanka/Projects/ML/basketball_shot_count/object_detection/sample_images')
    image_dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)

    image_to_show = 0
    for i, data in enumerate(image_dataloader):
        if i == image_to_show:
            file_names, images, scales, paddings = data

            # Image that is fed into the YOLO v3 object detection network
            # pil_image = transforms.ToPILImage()(images[0])
            # pil_image.show()

            detections = net.detect(images, scales, paddings)
            img = Image.open(file_names[0])
            net.draw_result(img, detections[0], show=True)
            break