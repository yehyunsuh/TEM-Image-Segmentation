"""
reference:
    reason of negative loss value:
        https://discuss.pytorch.org/t/bcewithlogitsloss-giving-negative-loss/123902
    torch clamp:
        https://aigong.tistory.com/178
"""

import torch
import numpy as np

from tqdm import tqdm
from log import log_results

import visualization as visualize


def train_function(args, DEVICE, model, loss_fn, optimizer, loader):
    total_loss = 0
    model.train()

    for image, label in tqdm(loader):
        ## change the label to have values between [0,1] 
        ## if not, loss will have negative values
        label = torch.clamp(label, min=0.0, max=1.0)

        image = image.float().to(device=DEVICE)
        label = label.to(device=DEVICE)
        predictions = model(image)

        # calculate log loss with pixel value
        loss = loss_fn(predictions, label)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return model, loss.item(), total_loss/len(loader)


def validate_function(args, DEVICE, model, epoch, loader):
    print("=====Starting Validation=====")
    model.eval()
    dice_score = 0

    with torch.no_grad():
        for idx, (image, label) in enumerate(tqdm(loader)):
            label = torch.clamp(label, min=0.0, max=1.0)

            image = image.float().to(DEVICE)
            label = label.to(DEVICE)
            
            prediction = model(image)

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()
            dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum())

            if epoch == 0:
                visualize.original_image(args, image, idx)
                visualize.original_image_with_label(args, image, label, idx)
            if epoch % int(args.epochs / 5) == 0 or args.epochs - 1 == epoch:
                visualize.original_image_with_prediction(args, image, prediction_binary, idx, epoch)

    dice = dice_score/len(loader)
    print(f"Dice score: {dice}")
    model.train()

    return model, dice

    
def train(
        args, DEVICE, model, loss_fn, optimizer, train_loader, val_loader
    ):
    count, pth_save_point = 0, 0
    best_loss = np.inf
    visualize.create_directories(args)
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")

        model, loss, mean_loss = train_function(
            args, DEVICE, model, loss_fn, optimizer, train_loader
        )
        model, dice = validate_function(
            args, DEVICE, model, epoch, val_loader
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }

        print("Current loss ", loss, mean_loss)
        if best_loss > mean_loss:
            print("=====New best model=====")
            best_loss, count = mean_loss, 0
        else:
            count += 1

        if args.wandb:
            log_results(
                mean_loss, dice
            )