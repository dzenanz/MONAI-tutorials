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
from pathlib import Path

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CSVSaver
from monai.transforms import AddChanneld, Compose, LoadImaged, Resized, ScaleIntensityd, ToTensord

from sklearn.metrics import confusion_matrix, classification_report

model_path = os.getcwd() + "/miqa01.pth"

def recursivelySearchImages(images, decisions, pathPrefix, kind):
    count = 0
    for path in Path(pathPrefix).rglob('*.nii.gz'):
        images.append(str(path))
        decisions.append(kind)
        count += 1
    print(f"{count} images in prefix {pathPrefix}")


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = []
    decisions = []
    recursivelySearchImages(images, decisions, os.getcwd()+'/NCANDA/unusable', 0)
    recursivelySearchImages(images, decisions, os.getcwd()+'/NCANDA/v01', 1)
    recursivelySearchImages(images, decisions, os.getcwd()+'/NCANDA/v03', 1)
    countTrain = len(images) # NCANDA is training group
    recursivelySearchImages(images, decisions, os.getcwd()+'/NCANDA/sri1', 1)
    recursivelySearchImages(images, decisions, os.getcwd()+'/NCANDA/sri0', 0)
    countVal = len(images) - countTrain # SRI is validation group
    print(f"{len(images)} images total in NCANDA")

    # 2 binary labels for scan classification: 1=good, 0=bad
    labels = np.asarray(decisions, dtype=np.int64)
    train_files = [{"img": img, "label": label} for img, label in zip(images[:countTrain], labels[:countTrain])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-countVal:], labels[-countVal:])]

    # Define transforms for image
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            ToTensord(keys=["img"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    print("Loading DenseNet121 classification model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path))

    print("Starting evaluation")
    y_pred = []
    y_true = []
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        saver = CSVSaver(output_dir=".")
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            val_outputs = model(val_images).argmax(dim=1)

            y_true.extend(val_data["label"].tolist())
            y_pred.extend(val_outputs.cpu().tolist())

            value = torch.eq(val_outputs, val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            saver.save_batch(val_outputs, val_data["img_meta_dict"])
            if metric_count % 20 == 0:
                print(f"Evaluated {metric_count}/{countVal}")
        metric = num_correct / metric_count
        print("evaluation metric:", metric)
        saver.finalize()

    print("confusion_matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
