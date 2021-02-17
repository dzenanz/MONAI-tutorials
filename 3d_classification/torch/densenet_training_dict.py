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
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import itkConfig

itkConfig.LazyLoading = False
import itk

import wandb

import monai
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadImaged, RandRotated, Resized, ScaleIntensityd, ToTensord

from sklearn.metrics import confusion_matrix, classification_report

model_path = os.getcwd() + "/miqa01.pth"
eCount = 0
nCount = 0


def getImageDimension(path):
    imageIO = itk.ImageIOFactory.CreateImageIO(path, itk.CommonEnums.IOFileMode_ReadMode)
    dim = (0, 0, 0)
    if imageIO is not None:
        try:
            imageIO.SetFileName(path)
            imageIO.ReadImageInformation()
            assert imageIO.GetNumberOfDimensions() == 3
            dim = (imageIO.GetDimensions(0), imageIO.GetDimensions(1), imageIO.GetDimensions(2))
        except RuntimeError:
            pass
    return dim


def recursivelySearchImages(images, decisions, pathPrefix, kind):
    count = 0
    for path in Path(pathPrefix).rglob('*.nii.gz'):
        images.append(str(path))
        decisions.append(kind)
        count += 1
    print(f"{count} images in prefix {pathPrefix}")


def constructPathFromCSVfields(participant_id, session_id, series_type, series_number, overall_qa_assessment):
    subNum = "sub-" + str(participant_id).zfill(6)
    sesNum = "ses-" + str(session_id)
    runNum = "run-" + str(series_number).zfill(3)
    sType = "PD"
    if series_type[0] == "T":  # not PD
        sType = series_type[0:2] + "w"
    if overall_qa_assessment < 6:
        sType = "BAD" + sType
    fileName = "P:/PREDICTHD_BIDS_DEFACE/" + subNum + "/" + sesNum + "/anat/" + \
               subNum + "_" + sesNum + "_" + runNum + "_" + sType + ".nii.gz"
    return fileName


def doesFileExist(fileName):
    my_file = Path(fileName)
    global eCount
    global nCount
    if my_file.is_file():
        # print(f"Exists: {fileName}")
        eCount += 1
        return True
    else:
        # print(f"Missing: {fileName}")
        nCount += 1
        return False


def readAndNormalizeDataFrame(tsvPath):
    df = pd.read_csv(tsvPath, sep='\t')
    df['file_path'] = df.apply(
        lambda row: constructPathFromCSVfields(row['participant_id'],
                                               row['session_id'],
                                               row['series_type'],
                                               row['series_number'],
                                               row['overall_qa_assessment'],
                                               ), axis=1)
    global eCount
    global nCount
    eCount = 0
    nCount = 0
    df['exists'] = df.apply(lambda row: doesFileExist(row['file_path']), axis=1)
    df['dimensions'] = df.apply(lambda row: getImageDimension(row['file_path']), axis=1)
    print(f"Existing files: {eCount}, non-existent files: {nCount}")
    return df


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    wandb.init(project="miqa_01", sync_tensorboard=True)
    config = wandb.config

    df = readAndNormalizeDataFrame(r'P:\PREDICTHD_BIDS_DEFACE\phenotype\bids_image_qc_information.tsv')
    print(df)
    df.to_csv(r'M:\Dev\zarr\bids_image_qc_information-my.csv', index=False)
    return

    images = []
    decisions = []
    for row in df.itertuples():
        if row.exists:
            images.append(row.file_path)
            decision = 0 if row.overall_qa_assessment < 6 else 1
            decisions.append(decision)

    countTrain = len(images)  # PredictHD is training group

    recursivelySearchImages(images, decisions, os.getcwd() + '/NCANDA/unusable', 0)
    recursivelySearchImages(images, decisions, os.getcwd() + '/NCANDA/v01', 1)
    recursivelySearchImages(images, decisions, os.getcwd() + '/NCANDA/v03', 1)
    countVal = len(images) - countTrain  # NCANDA is validation group
    print(f"{len(images)} images total")

    # check image size distribution
    sizes = {}
    for img in images:
        size = getImageDimension(img)
        if not size in sizes:
            sizes[size] = 1
        else:
            sizes[size] += 1
    print("Image size distribution:\n", sizes)

    # 2 binary labels for scan classification: 1=good, 0=bad
    labels = np.asarray(decisions, dtype=np.int64)
    train_files = [{"img": img, "label": label} for img, label in zip(images[:countTrain], labels[:countTrain])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-countVal:], labels[-countVal:])]

    # TODO: shuffle train_files

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            RandRotated(keys=["img"], prob=0.8, range_x=5, range_y=5, range_z=5),
            ToTensord(keys=["img"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            ToTensord(keys=["img"]),
        ]
    )

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # calculate class weights
    goodCount = np.sum(labels[:countTrain])
    badCount = countTrain - goodCount
    weightsArray = [goodCount / countTrain, badCount / countTrain]
    print(f"badCount: {badCount}, goodCount: {goodCount}, weightsArray: {weightsArray}")
    classWeights = torch.tensor(weightsArray, dtype=torch.float).to(device)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    if (os.path.exists(model_path)):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded NN model from file '{model_path}'")
    else:
        print("Training NN from scratch")

    loss_function = torch.nn.CrossEntropyLoss(weight=classWeights)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    config.learning_rate = 1e-5
    wandb.watch(model)

    # start a typical PyTorch training
    num_epochs = 20
    val_interval = 4
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter(log_dir=wandb.run.dir)
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
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
            wandb.log({"train_loss": loss.item()})
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        wandb.log({f"epoch average loss": epoch_loss})

        if (epoch + 1) % val_interval == 0:
            print("Starting evaluation")
            model.eval()
            y_pred = []
            y_true = []
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_images).argmax(dim=1)

                    y_true.extend(val_labels.cpu().tolist())
                    y_pred.extend(val_outputs.cpu().tolist())

                    num_correct += val_outputs.sum().item()
                    metric_count += len(val_outputs)
                    if metric_count % 20 == 0:
                        print(f"Evaluated {metric_count}/{countVal}")

                print("confusion_matrix:")
                print(confusion_matrix(y_true, y_pred))
                print(classification_report(y_true, y_pred))

                acc_metric = num_correct / metric_count
                auc_metric = compute_roc_auc(torch.as_tensor(y_pred), torch.as_tensor(y_true),
                                             average=monai.utils.Average.WEIGHTED)
                if auc_metric > best_metric:
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_path)
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'miqa01.pt'))
                    print("saved new best metric model")
                else:
                    epochSuffix = ".epoch" + str(epoch + 1)
                    torch.save(model.state_dict(), model_path + epochSuffix)
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'miqa01.pt' + epochSuffix))
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
                writer.add_scalar("val_AUC", auc_metric, epoch + 1)
                wandb.log({"val_accuracy": acc_metric})
                wandb.log({"val_AUC": auc_metric})

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
