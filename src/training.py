# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Training code
below code is adopted from https://github.com/harshatejas/pytorch_custom_object_detection
"""

# pylint: disable=[E1101,C0301,E0401,R0913,R0914,C0415]


import os
import argparse
import itertools
import time
import torch

from utils import VOCDataset, create_label_txt, get_model, \
    get_transforms, collate_fn, train_one_epoch, evaluate_model


def run_training(model, data_loader, data_loader_valid, num_epochs, optimizer, device, output_dir):
    """Train the model with the data passed"""
    # Learning rate scheduler decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    avg_time = []
    for epoch in range(num_epochs):
        start_time = time.time()
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        print('trained_model')
        end_time = time.time()
        avg_time.append(end_time - start_time)
        lr_scheduler.step()
        evaluate_model(model, data_loader_valid, device)

    print("Total Training Time Taken ==> ", sum(avg_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model state
    torch.save(model.state_dict(), os.path.join(output_dir, "vpd_model"))
    # to save model
    # torch.save(model, "model.pt")


def run_hyperparams(model, data_loader, data_loader_valid, device, output_dir, num_epochs=2):
    """run_hyperparameter function takes inputs of number of
    cross validation datasets
    Time taken for each param combination
    Total time taken for Hyperparameter tuning"""
    # hyperparams considered for tuning DL arch
    params = [p for p in model.parameters() if p.requires_grad]

    options = {
        "lr": [0.0001, 0.005],
        "opt": [0, 1]}

    # Replicating GridsearchCV functionality for params generation
    keys = options.keys()
    values = (options[key] for key in keys)
    p_combinations = []
    for combination in itertools.product(*values):
        if len(combination) > 0:
            p_combinations.append(combination)

    print("Total number of fits = ", len(p_combinations) * num_epochs)
    print("Take Break!!!\nThis will take time!")

    avg_time = []
    best_fit = {"combination": 0, "best_loss": 100, "optimizer": 'SGD'}

    for combination in p_combinations:
        learning_r, opt = combination
        if opt:
            optimizer = torch.optim.Adam(params, lr=learning_r)
        else:
            optimizer = torch.optim.SGD(params, lr=learning_r, momentum=0.9, weight_decay=0.0005)

        # Learning rate scheduler decreases the learning rate by 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epochs):

            start_time = time.time()
            curr_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
            end_time = time.time()
            avg_time.append(end_time - start_time)
            lr_scheduler.step()
            evaluate_model(model, data_loader_valid, device)
            if best_fit["best_loss"] > curr_loss:
                best_fit["best_loss"] = curr_loss
                best_fit["combination"] = combination
                best_fit["optimizer"] = optimizer
                print("hyper epoch", epoch)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save the model state
                torch.save(model.state_dict(), os.path.join(output_dir, "vpd_best_model"))
                # to save model
                # torch.save(model, "model.pt")

    print("Total Tuning Time Taken ==> ", sum(avg_time))
    print("Best parameters ==> ", best_fit)


def main():
    """main function """
    # Hyper-parameters
    train_batch_size = 8  # Train batch size
    valid_batch_size = 8  # Test batch size
    num_classes = 8  # Number of classes
    learning_rate = 0.005  # Learning rate
    # num_epochs = 10  # Number of epochs

    # ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        required=False,
                        default='data/',
                        help='dataset path which consists of train and valid folders')
    parser.add_argument('-c',
                        '--checkpoint_path',
                        type=str,
                        required=False,
                        default=None,
                        help='use absolute path to load the model for tuning, default is None')
    parser.add_argument('-o',
                        '--output_model_path',
                        type=str,
                        required=False,
                        default='models/stock',
                        help='output file name without extension to save the model')
    parser.add_argument('-hy',
                        '--hyperparameter_tuning',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for hyperparameter tuning , default is 0')
    parser.add_argument('-ep',
                        '--number_of_epochs',
                        type=int,
                        required=False,
                        default=10,
                        help='Number of epochs to train the model , default is 10')
    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for enabling intel pytorch optimizations, default is 0')

    flags = parser.parse_args()

    data_folder = flags.data_path
    intel_flag = flags.intel
    hyperparameter = flags.hyperparameter_tuning
    output_dir = flags.output_model_path
    checkpoint_path = flags.checkpoint_path
    num_epochs = flags.number_of_epochs

    # Setting up the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    labels_dict = create_label_txt()

    # Define train and test dataset
    dataset = VOCDataset(dataset_dir=data_folder + "/train/",
                         labels_dict=labels_dict, transforms=get_transforms(train=True))
    dataset_valid = VOCDataset(dataset_dir=data_folder + "/valid/",
                               labels_dict=labels_dict, transforms=get_transforms(train=False))

    # Split the dataset into train and test
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    indices_valid = torch.randperm(len(dataset_valid)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    dataset_valid = torch.utils.data.Subset(dataset_valid, indices_valid)

    # Define train and test dataloaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=1, collate_fn=collate_fn)

    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=valid_batch_size, shuffle=False,
                                                    num_workers=1, collate_fn=collate_fn)

    print(
        f"We have: {len(indices) + len(indices_valid)} images in the dataset, {len(dataset)} are "
        f"training images and {len(dataset_valid)} are test images")

    # Get the model using helper function
    model = get_model(num_classes)
    model.to(device=device)
    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    if intel_flag:
        import intel_extension_for_pytorch as ipex
        model = model.to(memory_format=torch.channels_last)
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print("IPEX optimization enabled")

    if hyperparameter:
        run_hyperparams(model, data_loader, data_loader_valid, device, output_dir)
    else:
        run_training(model, data_loader, data_loader_valid, num_epochs, optimizer, device, output_dir)


if __name__ == "__main__":
    main()
