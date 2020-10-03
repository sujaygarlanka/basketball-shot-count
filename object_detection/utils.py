import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as TF
import numbers
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import NUM_CLASSES, EPSILON

def cxcywh_to_xywh(bbox):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox


def xywh_to_cxcywh(bbox):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox

def untransform_bboxes(bboxes, scale, padding):
    """transform the bounding box from the scaled image back to the unscaled image."""
    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    # x, y, w, h = bbs
    x /= scale
    y /= scale
    w /= scale
    h /= scale
    x -= padding[0]
    y -= padding[1]
    return bboxes

def post_process(results_raw, nms, conf_thres, nms_thres):
    results = []
    for idx, result_raw in enumerate(results_raw):
        bboxes = result_raw[..., :4]
        scores = result_raw[..., 4]
        classes = result_raw[..., 5:]
        if nms:
            bboxes, scores, classes = \
            non_max_suppression(bboxes, scores, classes,
                                num_classes=NUM_CLASSES,
                                center=True,
                                conf_thres=conf_thres,
                                nms_thres=nms_thres)
        result = torch.cat((bboxes, scores.view((-1, 1)), classes), dim=1)
        results.append(result)
    return torch.stack(results)


def non_max_suppression(bboxes, scores, classes, num_classes, conf_thres=0.8, nms_thres=0.4, center=False):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        bboxes: (tensor) The location predictions for the img, Shape: [num_priors,4].
        scores: (tensor) The class prediction scores for the img, Shape:[num_priors].
        classes: (tensor) The label (non-one-hot) representation of the classes of the objects,
          Shape: [num_priors].
        num_classes: (int) The number of all the classes.
        conf_thres: (float) Threshold where all the detections below this value will be ignored.
        nms_thres: (float) The overlap thresh for suppressing unnecessary boxes.
        center: (boolean) Whether the bboxes format is cxcywh or xywh.
    Return:
        The indices of the kept boxes with respect to num_priors, and they are always in xywh format.
    """

    # make sure bboxes and scores have the same 0th dimension
    assert bboxes.shape[0] == scores.shape[0] == classes.shape[0]
    num_prior = bboxes.shape[0]

    # if no objects, return raw result
    if num_prior == 0:
        return bboxes, scores, classes

    # threshold out low confidence detection
    conf_index = torch.nonzero(torch.ge(scores, conf_thres), as_tuple=False).squeeze()

    bboxes = bboxes.index_select(0, conf_index)
    scores = scores.index_select(0, conf_index)
    classes = classes.index_select(0, conf_index)

    bboxes_result = []
    scores_result = []
    classes_result = []

    while bboxes.size(0) != 0:
        largest_index = torch.argmax(scores)
        current_bbox = bboxes[largest_index, :]
        bboxes_result.append(current_bbox)
        scores_result.append(scores[largest_index])
        classes_result.append(classes[largest_index])

        ious = iou_one_to_many(current_bbox, bboxes)
        valid_bbox_indices = torch.nonzero(torch.lt(ious, nms_thres), as_tuple=False).squeeze()
        
        bboxes = bboxes.index_select(0, valid_bbox_indices)
        scores = scores.index_select(0, valid_bbox_indices)
        classes = classes.index_select(0, valid_bbox_indices)

    return torch.stack(bboxes_result), torch.stack(scores_result), torch.stack(classes_result)

def iou_one_to_many(bbox1, bboxes2, center=False):
    """Calculate IOU for one bbox with another group of bboxes.
    If center is false, then they should all in xywh format.
    Else, they should all be in cxcywh format"""
    x1, y1, w1, h1 = bbox1
    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]
    if center:
        x1 = x1 - w1 / 2
        y1 = y1 - h1 / 2
        x2 = x2 - w2 / 2
        y2 = y2 - h2 / 2
    area1 = w1 * h1
    area2 = w2 * h2
    right1 = x1 + w1
    right2 = x2 + w2
    bottom1 = y1 + h1
    bottom2 = y2 + h2
    w_intersect = (torch.min(right1, right2) - torch.max(x1, x2)).clamp(min=0)
    h_intersect = (torch.min(bottom1, bottom2) - torch.max(y1, y2)).clamp(min=0)
    area_intersect = w_intersect * h_intersect
    iou_ = area_intersect / (area1 + area2 - area_intersect + EPSILON) #add epsilon to avoid NaN
    return iou_


# def convert_videos_to_images(dirpath_for_videos, dirpath_for_images):
#     num_frames = 16
#     vidcap = cv2.VideoCapture(video_path)
#     frames = []
#     # extract frames
#     while True:
#         success, image = vidcap.read()
#         if not success:
#             break
#         frames.append(image)
#     # downsample if desired and necessary
#     if num_frames < len(frames):
#         skip = len(frames) // num_frames
#         for i in range(0, len(frames), skip):
#             pass
#     return

def default_transform_fn(img_size):
    return ComposeWithLabel([PadToSquareWithLabel(fill=(127, 127, 127)),
                             ResizeWithLabel(img_size),
                             transforms.ToTensor()])

class ComposeWithLabel(transforms.Compose):

    def __call__(self, img, label=None):
        import inspect
        for t in self.transforms:
            num_param = len(inspect.signature(t).parameters)
            if num_param == 2:
                img, label = t(img, label)
            elif num_param == 1:
                img = t(img)
        return img, label

class ResizeWithLabel(transforms.Resize):

    def __init__(self, size, interpolation=Image.BILINEAR):
        super(ResizeWithLabel, self).__init__(size, interpolation)

    def __call__(self, img, label=None):
        w_old, h_old = img.size
        img = super(ResizeWithLabel, self).__call__(img)
        w_new, h_new = img.size
        if label is None:
            return img, label
        scale_w = w_new / w_old
        scale_h = h_new / h_old
        label[..., 0] *= scale_w
        label[..., 1] *= scale_h
        label[..., 2] *= scale_w
        label[..., 3] *= scale_h
        return img, label

class PadToSquareWithLabel(object):
    """Pad to square the given PIL Image with label.
    Args:
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def _get_padding(w, h):
        """Generate the size of the padding given the size of the image,
        such that the padded image will be square.
        Args:
            h (int): the height of the image.
            w (int): the width of the image.
        Return:
            A tuple of size 4 indicating the size of the padding in 4 directions:
            left, top, right, bottom. This is to match torchvision.transforms.Pad's parameters.
            For details, see:
                https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Pad
            """
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        return (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)

    def __call__(self, img, label=None):
        w, h = img.size
        padding = self._get_padding(w, h)
        img = TF.pad(img, padding, self.fill, self.padding_mode)
        if label is None:
            return img, label
        label[..., 0] += padding[0]
        label[..., 1] += padding[1]
        return img, label
