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
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadNiftid, RandRotate90d, Resized, ScaleIntensityd, ToTensord


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = []
    decisions = []
    with open('SRI_Sessions/session_ann_pretty.json') as json_file:
        data = json.load(json_file)
        for s in data['scans']:
            decision = int(s['decision'])
            path = os.getcwd()+"/SRI_Sessions/scanroot"+data['data_root']+s['path']+"/"            
            filenames = sorted(s['volumes'].keys())
            for f in filenames:
                images.append(path+f+".nii.gz")
                decisions.append(decision)

    # 2 binary labels for scan classification: 1=good, 0=bad
    labels = np.asarray(decisions, dtype=np.int64)
    train_files = [{"img": img, "label": label} for img, label in zip(images[:10], labels[:10])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-10:], labels[-10:])]

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadNiftid(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
            ToTensord(keys=["img"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadNiftid(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            ToTensord(keys=["img"]),
        ]
    )

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
