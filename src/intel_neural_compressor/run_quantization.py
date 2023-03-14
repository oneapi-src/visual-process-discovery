# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
INC QUANTIZATION model saving
"""

# pylint: disable=[E1101,C0301,E0401,R0914,R0205,C0103,R0903,C0413,E0102]
# flake8: noqa = E402

import sys
import argparse
import os
sys.path.insert(0, './src')
import numpy as np
import torch
from neural_compressor.data import Dataset
from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig
from utils import VOCDataset, get_transforms, get_model, collate_fn, evaluate_model


class Dataset:
    """Creating Dataset class for getting Image and labels"""

    def __init__(self, dataloader, batch_s, device):
        self.dataloader = dataloader
        self.batch_size = batch_s
        self.im = None
        cnt = 0
        for _, (images, targets) in enumerate(self.dataloader):
            self.im = np.array(list(np.array(image.to(device)) for image in images))
            self.im = torch.Tensor(self.im)
            self.targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # self.targets = self.targets[0]['boxes']
            # self.targets[0]["bbox"] = np.array([10,25,300,200])
            cnt += 1
            if cnt > self.batch_size:
                break

    def __getitem__(self, index):
        self.targets[index]["boxes"] = torch.Tensor([10, 25, 300, 200])
        self.targets[index]["labels"] = torch.Tensor([1])
        self.targets[index]['area'] = torch.Tensor([6230])
        self.targets[index]['iscrowd'] = torch.Tensor([0])
        return self.im[index], [self.targets[index]]

    def __len__(self):
        return self.batch_size  # len(self.dataloader) #len(self.test_images)

    def run_evaluation(self, pt_model):
        """Evaluating the model with test data and returning the metrics as mAp """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        metrics = evaluate_model(pt_model, self.dataloader, device=device)
        return metrics


def main():
    """ Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--outpath',
                        type=str,
                        required=False,
                        default='./models/inc_compressed_model/',
                        help='absolute path to save quantized model. By default it '
                             'will be saved in "./inc_compressed_model/output" folder')
    parser.add_argument('-d',
                        '--datapath',
                        type=str,
                        required=False,
                        default='data/',
                        help='Path to test dataset folder')
    parser.add_argument('-c',
                        '--checkpoint_path',
                        type=str,
                        required=False,
                        default=None,
                        help='path to load the quantization model')

    # Command line Arguments
    flags = parser.parse_args()
    out_path = flags.outpath
    data_path = flags.datapath
    test_batch_size = 1
    checkpoint_path = flags.checkpoint_path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    labels_dict = {0: 'button', 1: 'heading', 2: 'link', 3: 'label', 4: 'text', 5: 'image', 6: 'iframe', 7: 'field'}
    num_classes = 8  # Number of classes

    # Load model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model = model.to(memory_format=torch.channels_last)

    dataset = VOCDataset(dataset_dir=os.path.join(data_path, 'test'),
                         labels_dict=labels_dict, transforms=get_transforms(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=True,
                                              num_workers=1, collate_fn=collate_fn)

    # Quantization
    config = PostTrainingQuantConfig()
    dataset = Dataset(data_loader, test_batch_size, device)
    calib_dataloader = DataLoader(framework='pytorch', dataset=dataset)
    eval_func = dataset.run_evaluation
    q_model = fit(model=model,
        conf=config,
        calib_dataloader=calib_dataloader,
        eval_func = eval_func)
    q_model.save(out_path)

    print("*" * 30)
    print("Successfully Quantized model and saved at :", out_path)
    print("*" * 30)


if __name__ == "__main__":
    main()
