import re
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import TensorDataset, DataLoader
from CocoDetectionDataset import CocoDetectionBoundingBox
from ImageDataset import ImageFolder
import numpy as np
from model import YoloNetV3
from utils import post_process, untransform_bboxes, cxcywh_to_xywh
from PIL import Image, ImageDraw, ImageFont



class ObjectDetection():

    def __init__(self):
        self.model = YoloNetV3()
        self.model.eval()
        self.load_weights('yolov3.weights', self.model)

    def detect(self, images):
        with torch.no_grad():
            detections = self.model(images)
        detections = post_process(detections, True, 0.8, 0.4)
        for detection, scale, padding in zip(detections, scales, paddings):
            detection[..., :4] = untransform_bboxes(detection[..., :4], scale, padding)
            cxcywh_to_xywh(detection)
        return detections

    def load_weights(self, weightfile, model):
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        header = torch.from_numpy(header)
        seen = header[3]

        ptr = 0
        weights = np.fromfile(fp, dtype = np.float32)
        state_dict = model.state_dict()
        state_dict_keys = list(state_dict.keys())
        i = 0
        while (i < len(state_dict_keys)):
            layer_name = state_dict_keys[i]
            # Finds convolutional layers with batch normalization
            if ('conv.weight' in layer_name):
                bn_bias_layer = state_dict[state_dict_keys[i+2]]
                bn_weight_layer = state_dict[state_dict_keys[i+1]]
                bn_running_mean_layer = state_dict[state_dict_keys[i+3]]
                bn_running_var_layer = state_dict[state_dict_keys[i+4]]

                #Get the number of weights of Batch Norm Layer
                num_bn_biases = bn_bias_layer.numel()

                #Load the weights
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                #Cast the loaded weights into dims of model weights. 
                bn_biases = bn_biases.view_as(bn_bias_layer.data)
                bn_weights = bn_weights.view_as(bn_weight_layer.data)
                bn_running_mean = bn_running_mean.view_as(bn_running_mean_layer.data)
                bn_running_var = bn_running_var.view_as(bn_running_var_layer.data)

                #Copy the data to model
                bn_bias_layer.data.copy_(bn_biases)
                bn_weight_layer.data.copy_(bn_weights)
                bn_running_mean_layer.copy_(bn_running_mean)
                bn_running_var_layer.copy_(bn_running_var)

                # Get the convolutional layer
                conv_weights_layer = state_dict[state_dict_keys[i]]
                # Load weights into the convolutional layer
                num_weights = conv_weights_layer.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv_weights_layer.data)
                conv_weights_layer.copy_(conv_weights)
                
                i += 5

            # Finds convolutional layers without batch normalization
            elif (bool(re.search("conv[0-9]+\.weight", layer_name))):
                # Get the bias layer
                bias_layer = state_dict[state_dict_keys[i+1]]
                # Load biases into the convolutional layer
                num_biases = bias_layer.numel()
                biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                ptr = ptr + num_biases
                biases = biases.view_as(bias_layer.data)
                bias_layer.copy_(biases)

                # Get the convolutional layer
                conv_weights_layer = state_dict[state_dict_keys[i]]
                # Load weights into the convolutional layer
                num_weights = conv_weights_layer.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv_weights_layer.data)
                conv_weights_layer.copy_(conv_weights)

                i += 2

            else:
                i += 1

    def draw_result(self, img, boxes, show=False, class_names = None):
        if isinstance(img, torch.Tensor):
            transform = ToPILImage()
            img = transform(img)
        draw = ImageDraw.ImageDraw(img)
        # show_class = (boxes.size(1) >= 6)
        # if show_class:
        #     assert isinstance(class_names, list)
        for box in boxes:
            x, y, w, h = box[:4]
            x2 = x + w
            y2 = y + h
            draw.rectangle([x, y, x2, y2], outline='white', width=3)
            # if show_class:
            #     class_id = int(box[5])
            #     class_name = class_names[class_id]
            #     font_size = 20
            #     class_font = ImageFont.truetype("../fonts/Roboto-Regular.ttf", font_size)
            #     text_size = draw.textsize(class_name, font=class_font)
            #     draw.rectangle([x, y-text_size[1], x + text_size[0], y], fill='white')
            #     draw.text([x, y-font_size], class_name, font=class_font, fill='black')
        if show:
            img.show()
        return img

if __name__ == "__main__":
    model = ObjectDetection()
    
    # custom_dataset = CocoDetectionBoundingBox('./coco/val2017', './coco/instances_val2017.json', img_size=416)
    image_dataset = ImageFolder('/Users/SujayGarlanka/Desktop/test')

    # custom_dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)
    image_dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)

    image_to_show = 0
    for i, data in enumerate(image_dataloader):
        if i == image_to_show:
            file_names, images, scales, paddings = data
            detections = model.detect(images)
            img = Image.open(file_names[0])
            model.draw_result(img, detections[0], show=True)
            # print(detections[0][0][])
            break