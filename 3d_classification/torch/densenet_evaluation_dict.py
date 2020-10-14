# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CSVSaver
from monai.transforms import AddChanneld, Compose, LoadNiftid, Resized, ScaleIntensityd, ToTensord

model_path = os.getcwd() + "/miqa01.pth"

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = []
    decisions = []
    with open('SRI_Sessions/session_ann_pretty.json') as json_file:
        data = json.load(json_file)
        for s in data['scans']:
            decision = int(s['decision'])
            decLen = len(decisions)
            if (decLen>1 and decisions[decLen-1]==1 and decision==0):
                print("zero at index", decLen)
            path = os.getcwd()+"/SRI_Sessions/scanroot"+data['data_root']+s['path']+"/"            
            filenames = sorted(s['volumes'].keys())
            for f in filenames:
                images.append(path+f+".nii.gz")
                decisions.append(decision)

    print("image count:", len(images))
    print(decisions)

    # 2 binary labels for scan classification: 1=good, 0=bad
    labels = np.asarray(decisions, dtype=np.int64)
    train_files = [{"img": img, "label": label} for img, label in zip(images[:500], labels[:500])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-500:], labels[-500:])]

    # Define transforms for image
    val_transforms = Compose(
        [
            LoadNiftid(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            ToTensord(keys=["img"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        saver = CSVSaver(output_dir=".")
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            val_outputs = model(val_images).argmax(dim=1)
            value = torch.eq(val_outputs, val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            saver.save_batch(val_outputs, val_data["img_meta_dict"])
        metric = num_correct / metric_count
        print("evaluation metric:", metric)
        saver.finalize()


if __name__ == "__main__":
    main()
