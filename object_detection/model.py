import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ANCHORS, NUM_ANCHORS_PER_SCALE, NUM_CLASSES, NUM_ATTRIB, LAST_LAYER_DIM

# Paper for architecture is here https://arxiv.org/pdf/1804.02767.pdf
# Heavily inspired by https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py

# Convolution Layer with Batch Norm and Leaky ReLU Activation function
class ConvLayer(nn.Module):

    def __init__(self, in_filters, out_filters, kernel_size, stride=1, lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_filters)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        x = self.conv(x)
        # print(x)
        x = self.bn(x)
        # print(x)
        x = self.lrelu(x)
        return x

class ResBlock(nn.Module):
    def __init__ (self, in_filters):
        super(ResBlock, self).__init__()
        half_in_filters = in_filters// 2
        self.conv1 = ConvLayer(in_filters, half_in_filters, 1)
        self.conv2 = ConvLayer(half_in_filters, in_filters, 3)

    def forward(self, x):
        input_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += input_x
        return x 

class DecodeLayer(nn.Module):

    def __init__(self, scale):
        super(DecodeLayer, self).__init__()

        if scale == 's':
            idx = (0, 1, 2)
            self.stride = 8
        elif scale == 'm':
            idx = (3, 4, 5)
            self.stride = 16
        elif scale == 'l':
            idx = (6, 7, 8)
            self.stride = 32
        self.anchors = torch.tensor([ANCHORS[i] for i in idx])

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)

        if self.training:
            output_raw = x.view(num_batch,
                                NUM_ANCHORS_PER_SCALE,
                                NUM_ATTRIB,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, NUM_ATTRIB)
            return output_raw
        else:
            prediction_raw = x.view(num_batch,
                                    NUM_ANCHORS_PER_SCALE,
                                    NUM_ATTRIB,
                                    num_grid,
                                    num_grid).permute(0, 1, 3, 4, 2).contiguous()

            self.anchors = self.anchors.float()
            # Calculate offsets for each grid
            grid_tensor = torch.arange(num_grid, dtype=torch.float).repeat(num_grid, 1)
            grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
            grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
            anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
            anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

            # Get outputs
            x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride # Center x
            y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride  # Center y
            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
            bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, NUM_CLASSES)  # Cls pred one-hot.

            output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
            return output

class DarkNet53(nn.Module):

    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv1 = ConvLayer(3, 32, 3)

        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.res_block1 = ResBlock(64)

        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.res_block2 = ResBlock(128)
        self.res_block3 = ResBlock(128)

        self.conv4 = ConvLayer(128, 256, 3, 2)
        self.res_block4 = ResBlock(256)
        self.res_block5 = ResBlock(256)
        self.res_block6 = ResBlock(256)
        self.res_block7 = ResBlock(256)
        self.res_block8 = ResBlock(256)
        self.res_block9 = ResBlock(256)
        self.res_block10 = ResBlock(256)
        self.res_block11 = ResBlock(256)

        self.conv5 = ConvLayer(256, 512, 3, 2)
        self.res_block12 = ResBlock(512)
        self.res_block13 = ResBlock(512)
        self.res_block14 = ResBlock(512)
        self.res_block15 = ResBlock(512)
        self.res_block16 = ResBlock(512)
        self.res_block17 = ResBlock(512)
        self.res_block18 = ResBlock(512)
        self.res_block19 = ResBlock(512)

        self.conv6 = ConvLayer(512, 1024, 3, 2)
        self.res_block20 = ResBlock(1024)
        self.res_block21 = ResBlock(1024)
        self.res_block22 = ResBlock(1024)
        self.res_block23 = ResBlock(1024)

    def forward(self, x):
        x = self.conv1(x)
        # return x
        # print(x)
        x = self.conv2(x)
        x = self.res_block1(x)

        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.conv4(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.res_block9(x)
        x = self.res_block10(x)
        out3 = self.res_block11(x)

        x = self.conv5(x)
        x = self.res_block12(x)
        x = self.res_block13(x)
        x = self.res_block14(x)
        x = self.res_block15(x)
        x = self.res_block16(x)
        x = self.res_block17(x)
        x = self.res_block18(x)
        out2 = self.res_block19(x)

        x = self.conv6(x)
        x = self.res_block20(x)
        x = self.res_block21(x)
        x = self.res_block22(x)
        out1 = self.res_block23(x)

        return out1, out2, out3

class YoloNetTail(nn.Module):
    
    def __init__(self):
        super(YoloNetTail, self).__init__()

        # Detection block 1
        self.conv1 = ConvLayer(1024, 512, 1)
        self.conv2 = ConvLayer(512, 1024, 3)
        self.conv3 = ConvLayer(1024, 512, 1)
        self.conv4 = ConvLayer(512, 1024, 3)
        self.conv5 = ConvLayer(1024, 512, 1)
        self.conv6 = ConvLayer(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, LAST_LAYER_DIM, 1, bias=True)
        self.decodeL = DecodeLayer('l')

        # Detection block 2
        self.conv8 = ConvLayer(512, 256, 1)
        self.conv9 = ConvLayer(768, 256, 1)
        self.conv10 = ConvLayer(256, 512, 3)
        self.conv11 = ConvLayer(512, 256, 1)
        self.conv12 = ConvLayer(256, 512, 3)
        self.conv13 = ConvLayer(512, 256, 1)
        self.conv14 = ConvLayer(256, 512, 3)
        self.conv15 = nn.Conv2d(512, LAST_LAYER_DIM, 1, bias=True)
        self.decodeM = DecodeLayer('m')

        # Detection block 3
        self.conv16 = ConvLayer(256, 128, 1)
        self.conv17 = ConvLayer(384, 128, 1)
        self.conv18 = ConvLayer(128, 256, 3)
        self.conv19 = ConvLayer(256, 128, 1)
        self.conv20 = ConvLayer(128, 256, 3)
        self.conv21 = ConvLayer(256, 128, 1)
        self.conv22 = ConvLayer(128, 256, 3)
        self.conv23 = nn.Conv2d(256, LAST_LAYER_DIM, 1, bias=True)
        self.decodeS = DecodeLayer('s')

    def forward(self, x1, x2, x3):
        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        branch1 = self.conv5(x)
        x = self.conv6(branch1)
        x = self.conv7(x)
        outL = self.decodeL(x)

        x = self.conv8(branch1)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, x2), 1)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        branch2 = self.conv13(x)
        x = self.conv14(branch2)
        x = self.conv15(x)
        outM = self.decodeM(x)

        x = self.conv16(branch2)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, x3), 1)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        outS = self.decodeS(x)

        return outL, outM, outS

    def yolo_last_layers(self):
        _layers = [self.yolo_tail.conv7,
                   self.yolo_tail.conv15,
                   self.yolo_tail.conv23]
        return _layers

    def yolo_last_two_layers(self):
        _layers = self.yolo_last_layers() + \
                  [self.yolo_tail.conv6,
                   self.yolo_tail.conv14,
                   self.yolo_tail.conv22]
        return _layers

    def yolo_last_three_layers(self):
        _layers = self.yolo_last_two_layers() + \
                  [self.yolo_tail.conv5,
                   self.yolo_tail.conv13,
                   self.yolo_tail.conv21]
        return _layers

    def yolo_tail_layers(self):
        _layers = [self.yolo_tail]
        return _layers

    def yolo_last_n_layers(self, n):
        try:
            n = int(n)
        except ValueError:
            pass
        if n == 1:
            return self.yolo_last_layers()
        elif n == 2:
            return self.yolo_last_two_layers()
        elif n == 3:
            return self.yolo_last_three_layers()
        elif n == 'tail':
            return self.yolo_tail_layers()
        else:
            raise ValueError("n>3 not defined")

class YoloNetV3(nn.Module):

    def __init__(self, nms=False, post=True):
        super(YoloNetV3, self).__init__()
        self.darknet = DarkNet53()
        self.yolo_tail = YoloNetTail()
        self.nms = nms

    def forward(self, x):
        x1, x2, x3 = self.darknet(x)
        out1, out2, out3 = self.yolo_tail(x1, x2, x3)
        out = torch.cat((out1, out2, out3), 1)
        return out



