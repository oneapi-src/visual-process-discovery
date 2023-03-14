# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluation code
below code is adopted from https://github.com/harshatejas/pytorch_custom_object_detection
"""
# pylint: disable=[E1101,C0301,E0401,C0415]
import os
import sys
import argparse
import torch
from utils import VOCDataset, create_label_txt, get_model, get_transforms, \
    collate_fn, evaluate_model


def main():
    """
    Main Function
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
    flags = parser.parse_args()

    data_folder = flags.dataset
    intel_flag = flags.intel
    checkpoint_path = flags.checkpoint_path

    # Setting up the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    labels_dict = create_label_txt()

    # Define valid dataset for inference
    dataset_test = VOCDataset(dataset_dir=os.path.join(data_folder, 'test'),
                              labels_dict=labels_dict, transforms=get_transforms(train=False))

    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False,
                                              num_workers=1, collate_fn=collate_fn)
    print(f"We have: {len(indices_test)} images in the dataset, {len(dataset_test)} are test images")

    # Get the model using helper function
    model = get_model(num_classes)
    model.to(device=device)
    if checkpoint_path is None:
        print("Please provide path to the Model weights!!!")
        sys.exit(0)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    if intel_flag:
        import intel_extension_for_pytorch as ipex
        model = model.to(memory_format=torch.channels_last)
        model = ipex.optimize(model)
        print("IPEX optimization enabled")

    evaluate_model(model, data_loader, device)


if __name__ == "__main__":
    main()
