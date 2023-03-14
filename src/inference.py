# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Inference code
below code is adopted from https://github.com/harshatejas/pytorch_custom_object_detection
"""
# pylint: disable=[E1101,C0301,E0401,R0914,C0415]

import os
import argparse
import time
import torch
from utils import VOCDataset, create_label_txt, get_model, get_transforms, \
    collate_fn, MetricLogger


def main():
    """
    Main Function code
    """
    # Hyper parameters
    num_classes = 8  # Number of classes

    # ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        required=False,
                        default='data/',
                        help='dataset path which consists of test folder')
    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for enabling intel pytorch optimizations, default is 0')
    parser.add_argument('-c',
                        '--checkpoint_path',
                        type=str,
                        required=False,
                        default=None,
                        help='use 1 for enabling intel pytorch optimizations, default is 0')
    parser.add_argument('-b',
                        '--test_batch_size',
                        type=int,
                        required=False,
                        default=1,
                        help='use different batch sizes, default is 1')
    flags = parser.parse_args()

    data_folder = flags.dataset
    intel_flag = flags.intel
    checkpoint_path = flags.checkpoint_path
    test_batch_size = flags.test_batch_size

    # Setting up the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    labels_dict = create_label_txt()

    # Define test dataset for inference
    dataset_test = VOCDataset(dataset_dir=os.path.join(data_folder, 'test'),
                              labels_dict=labels_dict, transforms=get_transforms(train=False))

    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False,
                                              num_workers=1, collate_fn=collate_fn)
    print(f"We have: {len(indices_test)} images in the dataset, {len(dataset_test)} are test images")

    # Get the model using helper function
    model = get_model(num_classes)

    model.to(device=device)

    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    if intel_flag:
        import intel_extension_for_pytorch as ipex
        model = model.to(memory_format=torch.channels_last)
        model = ipex.optimize(model)
        print("IPEX optimization enabled")

    metric_logger = MetricLogger(delimiter="  ")

    cnt = 0
    header = 'Sample: [{}]'.format(cnt)
    avg_time = 0
    for images, _ in metric_logger.log_every(data_loader, 1, header):
        header = 'Sample: [{}]'.format(cnt)
        cnt += 1
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            start_time = time.time()
            model(images)
            avg_time += time.time() - start_time
        if cnt > 10:
            break

    print("Time Taken for inference using ", test_batch_size, " is ===> ", avg_time / cnt)


if __name__ == "__main__":
    main()
