import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import json
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import itk

import wandb

import monai
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets.regressor import Regressor
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadImaged, RandSpatialCropd, ScaleIntensityd, ToTensord

from sklearn.metrics import confusion_matrix, classification_report

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


class tiled_classifier(monai.networks.nets.Classifier):
    def forward(self, inputs):
        return super().forward(inputs)


def evaluateModel(model, dataLoader, device, writer, epoch, setName):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for val_data in dataLoader:
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            val_outputs = model(val_images).argmax(dim=1)

            y_true.extend(val_labels.cpu().tolist())
            y_pred.extend(val_outputs.cpu().tolist())

            num_correct += val_outputs.sum().item()
            metric_count += len(val_outputs)
            print('.', end='')
            if metric_count % 60 == 0:
                print("")

        print("\n" + setName + "_confusion_matrix:")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))

        acc_metric = num_correct / metric_count
        auc_metric = compute_roc_auc(torch.as_tensor(y_pred), torch.as_tensor(y_true),
                                     average=monai.utils.Average.WEIGHTED)
        writer.add_scalar(setName + "_accuracy", acc_metric, epoch + 1)
        writer.add_scalar(setName + "_AUC", auc_metric, epoch + 1)
        wandb.log({setName + "_accuracy": acc_metric})
        wandb.log({setName + "_AUC": auc_metric})
        return auc_metric, acc_metric


def trainAndSaveModel(df, countTrain, savePath, num_epochs, val_interval, evaluationOnly):
    images = []
    decisions = []
    sizes = {}
    for row in df.itertuples():
        if row.exists:
            images.append(row.file_path)
            decision = 0 if row.overall_qa_assessment < 6 else 1
            decisions.append(decision)

            size = row.dimensions
            if not size in sizes:
                sizes[size] = 1
            else:
                sizes[size] += 1

    # 2 binary labels for scan classification: 1=good, 0=bad
    labels = np.asarray(decisions, dtype=np.int64)
    countVal = df.shape[0] - countTrain
    train_files = [{"img": img, "label": label} for img, label in zip(images[:countTrain], labels[:countTrain])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-countVal:], labels[-countVal:])]

    # TODO: shuffle train_files (if not already done)

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            RandSpatialCropd(keys=["img"], roi_size=(128, 48, 48), random_center=True, random_size=False),
            ToTensord(keys=["img"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            RandSpatialCropd(keys=["img"], roi_size=(128, 48, 48), random_center=True, random_size=False),
            ToTensord(keys=["img"]),
        ]
    )

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=4, num_workers=2, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(f'Single input\'s shape: {check_data["img"].shape}, label: {check_data["label"]}')

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # calculate class weights
    goodCount = np.sum(labels[:countTrain])
    badCount = countTrain - goodCount
    weightsArray = [goodCount / countTrain, badCount / countTrain]
    print(f"badCount: {badCount}, goodCount: {goodCount}, weightsArray: {weightsArray}")
    classWeights = torch.tensor(weightsArray, dtype=torch.float).to(device)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    # model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model = tiled_classifier(in_shape=(1, 128, 48, 48), classes=2,
                             channels=(2, 4, 8, 16),
                             strides=(2, 2, 2, 2,)).to(
        device)

    if os.path.exists(savePath) and evaluationOnly:
        model.load_state_dict(torch.load(savePath))
        print(f"Loaded NN model from file '{savePath}'")
    else:
        print("Training NN from scratch")

    loss_function = torch.nn.CrossEntropyLoss(weight=classWeights)
    wandb.config.learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), wandb.config.learning_rate)
    wandb.watch(model)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter(log_dir=wandb.run.dir)

    if evaluationOnly:
        auc_metric, acc_metric = evaluateModel(model, val_loader, device, writer, 0, "eval")
        return sizes

    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"epoch_len: {epoch_len}")
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
            print(f"{step}:{loss.item():.4f}", end=' ')
            if step % 10 == 0:
                print("")  # new line
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            wandb.log({"train_loss": loss.item()})
        epoch_loss /= step
        print(f"\nepoch {epoch + 1} average loss: {epoch_loss:.4f}")
        wandb.log({f"epoch average loss": epoch_loss})

        if (epoch + 1) % val_interval == 0:
            print("Evaluating on validation set")
            auc_metric, acc_metric = evaluateModel(model, val_loader, device, writer, epoch, "val")

            if auc_metric >= best_metric:
                best_metric = auc_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), savePath)
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'miqa01.pt'))
                print("saved new best metric model")
            else:
                epochSuffix = ".epoch" + str(epoch + 1)
                torch.save(model.state_dict(), savePath + epochSuffix)
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'miqa01.pt' + epochSuffix))
            print(
                "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                    epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                )
            )

            print("Evaluating on training set")
            evaluateModel(model, train_loader, device, writer, epoch, "train")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    return sizes


def main(valdationFold, evaluationOnly):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    wandb.init(project="miqa_01", sync_tensorboard=True)

    folds = []
    for f in range(3):
        folds.append(pd.read_csv(f"M:/Dev/zarr/T1_fold{f}.csv"))

    df = pd.concat(folds)
    print(df)

    print(f"Using fold {valdationFold} for validation")
    vf = folds.pop(valdationFold)
    folds.append(vf)
    df = pd.concat(folds)
    countTrain = df.shape[0] - vf.shape[0]
    model_path = os.getcwd() + f"/miqa01-val{valdationFold}.pth"
    sizes = trainAndSaveModel(df, countTrain, savePath=model_path, num_epochs=15, val_interval=4,
                              evaluationOnly=evaluationOnly)

    print("Image size distribution:\n", sizes)


if __name__ == "__main__":
    # df = readAndNormalizeDataFrame(r'P:\PREDICTHD_BIDS_DEFACE\phenotype\bids_image_qc_information.tsv')
    # print(df)
    # df.to_csv(r'M:\Dev\zarr\bids_image_qc_information-my.csv', index=False)
    # return

    fold = 2
    if len(sys.argv) > 1:
        fold = int(sys.argv[1])

    evaluationOnly = False
    if len(sys.argv) > 2:
        evaluationOnly = (int(sys.argv[2]) != 0)

    main(fold, evaluationOnly)
