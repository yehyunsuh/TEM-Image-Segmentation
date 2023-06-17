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
        label = torch.clamp(label, max=1.0)

        image = image.float().to(device=DEVICE)
        label = label.to(device=DEVICE)
        predictions = torch.sigmoid(model(image))

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
            image = image.float().to(DEVICE)
            label = label.to(DEVICE)
            
            predictions = torch.sigmoid(model(image))

            ## make predictions to be 0. or 1.
            predictions_binary = (predictions > 0.5).float()
            dice_score += (2 * (predictions_binary * label).sum()) / ((predictions_binary + label).sum())

            if epoch == 0:
                visualize.original_image(args, image, idx)
                visualize.original_image_with_label(args, image, label, idx)
            if epoch % 50 == 0 or args.epochs - 1 == epoch:
                visualize.original_image_with_prediction(args, image, predictions_binary, idx, epoch)

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

        # if not args.no_image_save:
        #     save_predictions_as_images(args, val_loader, model, epoch, highest_probability_pixels_list, label_list_total, device=DEVICE)
        
        # if pth_save_point % 5 == 0: 
        #     torch.save(checkpoint, f"./results/UNet_Epoch_{epoch}.pth")
        # pth_save_point += 1

        # if sum(angle_list)/(len(val_loader)*3) < best_angle_mean:
        #     best_angle_mean = sum(angle_list)/(len(val_loader)*3)
        #     torch.save(checkpoint, f'./plot_results/{args.wandb_name}/results/{args.wandb_name}_best.pth')
        # if epoch == args.epochs - 1:
        #     torch.save(checkpoint, f'./plot_results/{args.wandb_name}/results/{args.wandb_name}.pth')
        #     box_plot(args, mse_list)
        #     if args.no_image_save:
        #         save_predictions_as_images(args, val_loader, model, epoch, highest_probability_pixels_list, label_list_total, device=DEVICE)

        # if highest_probability_mse_total/len(val_loader) < best_rmse_mean:
        #     best_rmse_mean = highest_probability_mse_total/len(val_loader)

        # print(f'pixel loss: {loss_pixel}, geometry loss: {loss_geometry}, angle loss: {loss_angle}')
        # print(f'best average rmse diff: {best_rmse_mean}, best average angle diff: {best_angle_mean}')

        if args.wandb:
            log_results(
                mean_loss, dice
            )

        # if args.patience and count == args.patience_threshold:
        #     print("Early Stopping")
        #     break