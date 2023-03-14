# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
utils code
below code is adopted from https://github.com/harshatejas/pytorch_custom_object_detection
"""
# pylint: disable=[E1101,C0301,E0401,R0913,R0914,R0402,C0103,R1705,R0205,C0200,R1721,C0116,W1514,C0209,R1736]
# flake8: noqa = E402

import sys
import math
from collections import defaultdict, deque
import datetime
import time
import errno
import os
from operator import itemgetter
import xml.etree.ElementTree as ElementTree
import numpy as np
import cv2
import torch
import torch.distributed as dist
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T


def check_boundary(width, height, xmin, ymin, xmax, ymax):
    """
    check_boundary
    """
    if int(ymin) <= 0:
        ymin = 1
    if int(xmin) <= 0:
        xmin = 1
    if int(xmin) >= width:
        xmin = 1021
    if int(xmax) >= width:
        xmax = 1022
    if int(ymax) >= height:
        ymax = 767
    if int(ymin) >= height:
        ymin = 766
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    if ymin == ymax:
        ymin, ymax = height-1, height
    if xmin == xmax:
        xmin, xmax = width-2, width-1
    return xmin, ymin, xmax, ymax


def IntersectBBox(bbox1, bbox2):
    """
    IntersectBBox
    """
    intersect_bbox = []
    if bbox2[0] >= bbox1[2] or bbox2[2] <= bbox1[0] or bbox2[1] >= bbox1[3] or bbox2[3] <= bbox1[1]:
        # return [0, 0, 0, 0], if there is no intersection
        return intersect_bbox
    else:
        intersect_bbox.append([max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
                               min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])])
    return intersect_bbox


def JaccardOverlap(bbox1, bbox2):
    """
    JaccardOverlap
    """
    intersect_bbox = IntersectBBox(bbox1, bbox2)
    if len(intersect_bbox) == 0:
        return 0
    else:
        intersect_width = int(intersect_bbox[0][2]) - int(intersect_bbox[0][0])
        intersect_height = int(intersect_bbox[0][3]) - int(intersect_bbox[0][1])
        if intersect_width and intersect_height > 0:
            intersect_size = float(intersect_width) * float(intersect_height)
            bbox1_size = float(bbox1[3] - bbox1[1]) * float(bbox1[2] - bbox1[0])
            bbox2_size = float(bbox2[3] - bbox2[1]) * float(bbox2[2] - bbox2[0])
            return float(intersect_size / float(bbox1_size + bbox2_size - intersect_size))
        else:
            return 0


def parse_one_annot(path, filename, labels_dict):
    """
    parse_one_annot
    """
    label_classes = list(labels_dict.values())
    # label_classes = ["button", "heading", "link", "label", "text", "image", "iframe", "field"]
    tree = ElementTree.parse(os.path.join(path, filename[:-3] + "xml"))
    root = tree.getroot()
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    list_with_all_boxes = []
    list_with_classes = []
    for boxes in root.iter('object'):
        name = boxes.find("name").text
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        xmin, ymin, xmax, ymax = check_boundary(width, height, xmin, ymin, xmax, ymax)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]

        list_with_all_boxes.append(list_with_single_boxes)
        list_with_classes.append(label_classes.index(str(name)))

    boxes_array = np.array(list_with_all_boxes)

    # Convert list to tuple
    classes = tuple(list_with_classes)
    return boxes_array, classes


class VOCDataset(torch.utils.data.Dataset):
    """ The dataset contains images of UI
        The dataset includes images of buttons, heading, link, label, text, image, iframe, field """

    def __init__(self, dataset_dir, labels_dict, transforms=None):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.labels_dict = labels_dict
        self.image_names = []
        dump = []

        pat = os.path.join(dataset_dir)
        if pat.endswith('/'):
            pat1 = pat[:-1]
        else:
            pat1 = pat

        with open(pat1 + ".txt", 'r') as fp:
            for line in fp:
                #sorts duplicate image names
                line = line.replace(' - Copy','') # removes 'Copy' from duplicate image names
                dump.append(line[:-1])
        for file in sorted(os.listdir(os.path.join(dataset_dir))):
            if file.endswith('.jpg') or file.endswith('.JPG'):
                if file in dump:
                    self.image_names.append(file)

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.image_names[index])
        image = cv2.imread(image_path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box_array, classes = parse_one_annot(self.dataset_dir, self.image_names[index], self.labels_dict)

        boxes = torch.as_tensor(box_array, dtype=torch.int64)

        labels = torch.tensor(classes, dtype=torch.int64)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.tensor(classes, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_names)


def create_label_txt():
    """
    create_label_txt
    """
    # Labels acquired from the dataset
    labels = ["button", "heading", "link", "label", "text", "image", "iframe", "field"]

    labels_dict = {}

    # Creat dictionary from array
    for index, label in enumerate(labels):
        labels_dict[index] = label

    # We need to create labels.txt and write labels dictionary into it
    with open('labels_vpd.txt', 'w') as f:
        f.write(str(labels_dict))

    return labels_dict


def get_model(num_classes):
    """
    # Load a pre-trained object detecting model (in this case faster-rcnn)
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transforms(train):
    """
    get_transforms
    """
    transforms = [T.ToTensor()]

    # Convert numpy image to PyTorch Tensor

    if train:
        # Data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        update
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        median
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    """
    MetricLogger
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        mb = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / mb))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))



def collate_fn(batch):
    """
    collate_fn
    """
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    warmup_lr_scheduler
    """

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    """
    mkdir
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def is_dist_avail_and_initialized():
    """
    is_dist_avail_and_initialized
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    get_world_size
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    get_rank
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    is_main_process
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    save_on_master
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    train_one_epoch
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger.loss.value


def _get_iou_types(model):
    """
    _get_iou_types
    """
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def cumTpFp(gtRects, detRects, label, overlapRatio):
    """
    cumTpFp
    # gtRect: label, xmin, ymin, xmax, ymax
    # gtRect: label, xmin, ymin, xmax, ymax, score
    # scores: scores for label
    """

    scores = detRects[:, :5]
    num_pos = len(gtRects)

    det_state = [(label, 0., 0, 1)] * len(detRects)

    for cnt in range(len(det_state)):
        det_state[cnt] = (label, scores[cnt], 0, 1)
    visited = [0] * num_pos

    if len(detRects) != len(scores):
        print("Num of scores does not match detection results!")
    for indexDet, _ in enumerate(detRects):
        iou_max = 0
        maxIndex = -1
        blockIdx = -1
        for indexGt, _ in enumerate(gtRects):
            iou = JaccardOverlap(detRects[indexDet], gtRects[indexGt])
            if iou > iou_max:
                iou_max = iou
                maxIndex = indexDet
                blockIdx = indexGt
        if iou_max >= overlapRatio and visited[blockIdx] == 0:
            det_state[maxIndex] = (label, scores[indexDet], 1, 0)
            visited[blockIdx] = 1

    return det_state, num_pos


def get_tp_fp(groundtruths, predictions, label, overlapRatio=0.5):
    """
    get_tp_fp
    """
    state_all = []
    tp = []
    fp = []
    all_num_pos = 0
    det_state, num_pos = cumTpFp(groundtruths, predictions, label, overlapRatio)
    all_num_pos += num_pos
    state_all += det_state

    for state in state_all:
        tp.append((state[1], state[2]))
        fp.append((state[1], state[3]))
    return tp, fp, all_num_pos


def CumSum_tp(tp, threshold=0.5):
    """
    CumSum_tp
    """
    tp = [tp]
    tp_copy = sorted(tp, key=itemgetter(0), reverse=True)
    cumsum = []
    cumPre = 0
    tp_th = 0
    tp_th_num = 0
    for index, _ in enumerate(tp_copy[0]):
        cumPre += (tp_copy[0][index][1])
        cumsum.append(cumPre)
        if tp_copy[0][index][0][0] > threshold:
            tp_th_num += 1
            if tp_copy[0][index][1] == 1:
                tp_th += 1
    tp_precision = float(tp_th) / float(tp_th_num)
    return cumsum, tp_th, tp_precision


def CumSum(fp, threshold=0.5):
    """
    CumSum
    """
    fp = [fp]
    fp_copy = sorted(fp, key=itemgetter(0), reverse=True)
    cumsum = []
    cumPre = 0
    fp_th = 0
    fp_th_num = 0
    for index, _ in enumerate(fp_copy[0]):
        cumPre += (fp_copy[0][index][1])
        cumsum.append(cumPre)
        if fp_copy[0][index][0][0] > threshold:
            fp_th_num += 1
            if fp_copy[0][index][1] == 1:  # false positive
                fp_th += 1
    fppw = float(fp_th) / float(fp_th_num)
    return cumsum, fp_th, fppw


def computeAp(tp, fp, all_num_pos):
    """
    computeAp
    """
    num = len(tp)
    prec = []
    rec = []
    fpr = []
    ap = 0
    if num == 0 or all_num_pos == 0:
        return prec, rec, 'inf', 'inf', ap

    tp_cumsum, tp_th, tp_precision = CumSum_tp(tp, threshold=0.5)
    fp_cumsum, fp_th, fppw = CumSum(fp, threshold=0.5)

    for i in range(num):
        prec.append(float(tp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))
        rec.append(float(tp_cumsum[i]) / float(all_num_pos))
        fpr.append(float(fp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))

    fppi = float(fp_th) / float(all_num_pos)

    # VOC2007 style for computing AP.
    max_precs = [0.] * 11
    start_idx = num - 1
    j = 10
    while j >= 0:
        i = start_idx
        while i >= 0:
            tmp = j / 10.0
            if rec[i] < tmp:
                start_idx = i
                if j > 0:
                    max_precs[j - 1] = max_precs[j]
                    break
            else:
                if max_precs[j] < prec[i]:
                    max_precs[j] = prec[i]
            i -= 1
        j -= 1
    for iji in range(11):
        ap += max_precs[iji] / 11.0

    # 计算 recall 和 precision
    recall = float(tp_th) / float(all_num_pos)
    precision = tp_precision
    return precision, recall, fppi, fppw, ap


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """
    evaluate_model
    """
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    class_map_temp = {0: 'button', 1: 'heading', 2: 'link', 3: 'label', 4: 'text', 5: 'image', 6: 'iframe', 7: 'field'}
    metric_config = {}
    for class_label in class_map_temp.values():
        metric_config[class_label] = {"Ap": 0, "precision": 0, "recall": 0, "FPPI": 0, "FPPW": 0}
    count = 0
    mAp = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        groundtruths = []
        for i, box in enumerate(targets[0]['boxes']):
            groundtruths.append([targets[0]['labels'][i], box[0], box[1], box[2], box[3]])
        predictions = []
        for i, box in enumerate(outputs[0]['boxes']):
            predictions.append([outputs[0]['labels'][i], box[0], box[1], box[2], box[3], outputs[0]['scores'][i]])
        predictions, groundtruths = np.array(predictions), np.array(groundtruths)
        if len(predictions) < 1:
            continue
        aps = 0
        count += 1
        for label in range(0, len(class_map_temp)):
            semantic_label = class_map_temp[label]
            tp, fp, all_num_pos = get_tp_fp(groundtruths, predictions, semantic_label, overlapRatio=0.5)
            precision, recall, fppi, fppw, ap = computeAp(tp, fp, all_num_pos)
            metric_config[semantic_label]["Ap"] += ap
            metric_config[semantic_label]["precision"] += precision
            metric_config[semantic_label]["recall"] += recall
            metric_config[semantic_label]["FPPI"] += fppi
            metric_config[semantic_label]["FPPW"] += fppw
            aps += ap
        mAp += aps / (len(class_map_temp) - 1)
    for semantic_label in class_map_temp.values():
        if count!=0:
            metric_config[semantic_label]["Ap"] = metric_config[semantic_label]["Ap"] / count
            metric_config[semantic_label]["precision"] = metric_config[semantic_label]["precision"] / count
            metric_config[semantic_label]["recall"] = metric_config[semantic_label]["recall"] / count
            metric_config[semantic_label]["FPPI"] = metric_config[semantic_label]["FPPI"] / count
            metric_config[semantic_label]["FPPW"] = metric_config[semantic_label]["FPPW"] / count
    print(metric_config)
    if not count:
        count = 1
    print("\nMAP: ", mAp / count)
    total_mAp = mAp / count
    return total_mAp
