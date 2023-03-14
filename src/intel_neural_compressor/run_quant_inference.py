# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
INC QUANTIZATION model saving
"""
# pylint: disable=[E1101,C0301,E0401,R0914,R0205,C0103,R0903,C0413]
# flake8: noqa = E402
import argparse
import os
import time
import sys
import numpy as np
import torch
from neural_compressor.utils.pytorch import load
import intel_extension_for_pytorch as ipex
sys.path.insert(0, './src')
from utils import VOCDataset, get_transforms, get_model, collate_fn, evaluate_model


labels_dict = {0: 'button', 1: 'heading', 2: 'link', 3: 'label', 4: 'text', 5: 'image', 6: 'iframe', 7: 'field'}


class Dataset:
    """Creating Dataset class for getting Image and labels"""

    def __init__(self, dataloader, batch_s):
        self.dataloader = dataloader
        self.batch_size = batch_s
        self.im = None
        cnt = 0
        for _, (images_in, targets_in) in enumerate(self.dataloader):
            self.im = np.array(list(np.array(image.to(device)) for image in images_in))
            self.im = torch.Tensor(self.im)
            self.targets = [{k: v.to(device) for k, v in t.items()} for t in targets_in]
            # self.targets = self.targets[0]['boxes']
            cnt += 1
            if cnt > self.batch_size:
                break

    def __getitem__(self, index):
        return self.im[index], self.targets[index]

    def __len__(self):
        return self.batch_size  # len(self.dataloader) #len(self.test_images)


# Define the command line arguments to input the Hyper parameters - batch size & Learning Rate
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        required=False,
                        default='data/',
                        help='path to the test data ')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=1,
                        help='batch size for the dataloader....default is 1')
    parser.add_argument('-qw',
                        '--quant_weights',
                        type=str,
                        required=False,
                        default="./models/inc_compressed_model/output",
                        help='Quantization Model Weights folder containing ".pt" format model')
    parser.add_argument('-eval',
                        '--eval_mode',
                        type=bool,
                        required=False,
                        default=False,
                        help='Enable evaluation mode to evaluate quantized model...default is False')

    # Command line Arguments
    flags = parser.parse_args()
    test_batch_size = flags.batch_size
    data_path = flags.data_path
    quant_weights = flags.quant_weights
    eval_mode = flags.eval_mode
    num_classes = 8

    # Load model
    model = get_model(num_classes)
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = VOCDataset(dataset_dir=os.path.join(data_path, 'test'),
                         labels_dict=labels_dict, transforms=get_transforms(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=True,
                                              num_workers=1, collate_fn=collate_fn)

    q_model = load(quant_weights, model)
    q_model = q_model.to(memory_format=torch.channels_last)
    q_model = ipex.optimize(q_model)
    print("IPEX optimization enabled")

    if eval_mode:
        evaluate_model(q_model, data_loader, device)
        sys.exit(0)

    # Timing Analysis
    INT_AVG_TIME = 0
    MAX_NUM_ITERATIONS = 10
    COUNT = 0

    with torch.no_grad():
        for _ in range(len(data_loader)):
            (im, _) = next(iter(data_loader))
            images = list(image.to(device) for image in im)
            # images = torch.Tensor(images)
            COUNT += 1
            if COUNT > MAX_NUM_ITERATIONS:
                break
            start_time = time.time()
            pred_q = q_model(images)
            pred_time_int = time.time() - start_time
            INT_AVG_TIME += pred_time_int

    print("Batch Size used here is ", test_batch_size)
    print("Average Inference Time Taken Int8 --> ", (INT_AVG_TIME / COUNT))
