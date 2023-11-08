"""
reference:
    reason of negative loss value:
        https://discuss.pytorch.org/t/bcewithlogitsloss-giving-negative-loss/123902
    torch clamp:
        https://aigong.tistory.com/178
"""

import torch
import torch.nn as nn
import numpy as np
import wandb
import pickle

from tqdm import tqdm
from log import log_results, make_dir

import visualization as visualize
import bfs as bfs


def train_function(args, DEVICE, model, loss_fn, optimizer, loader):
    total_loss = 0
    model.train()
    
    if args.distribution_loss and args.dist_loss:
        print("<<< Use distribution loss >>>")
    
    for image, label, w_mask, num_units, num_pixels in tqdm(loader):
        ## change the label to have values between [0,1] 
        ## if not, loss will have negative values
        label = torch.clamp(label, min=0.0, max=1.0)

        image = image.float().to(device=DEVICE)
        label = label.to(device=DEVICE)
        w_mask = w_mask.to(device=DEVICE)
        
        predictions = model(image)

        if args.weight_on_border:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.loss_weight], device=DEVICE), weight=w_mask)
        
        # calculate log loss with pixel value
        loss = loss_fn(predictions, label)
        
        if args.distribution_loss and args.dist_loss:
            pred_num = []
            pred_pixel = []
            predictions_binary = (predictions > args.prediction_threshold).float()
            
            for i in range(len(predictions_binary)):
                pred = predictions_binary[i][0].detach().cpu().numpy()
                pred = pred*255
                v_img = np.zeros((args.image_resize,args.image_resize), np.uint8)
                
                _, units = bfs.bfs(v_img, args.image_resize, args.image_resize, pred)
                
                if args.pore:
                    num, pixel = bfs.padding_pore(args, units)
                else:
                    num, pixel = bfs.padding_np(args, units)
                
                pred_num.append(num)
                pred_pixel.append(pixel)
            
            pred_pixel = np.array(pred_pixel)
            pred_pixel = pred_pixel*int((2048/args.image_resize)**2)
            
            num_pixels = num_pixels.numpy()
            num_units = num_units.numpy()
            
            t_pixels = num_pixels.shape[0]*num_pixels.shape[1]
            
            num_diff = (num_units-pred_num).sum()**2/len(num_units)
            pix_diff = (num_pixels-pred_pixel).sum()**2/t_pixels
            
            distribution_loss = args.weight_num*(num_diff**(1/2)) + args.weight_pixel*(pix_diff**(1/2))
            
            loss += distribution_loss

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
        for idx, (image, label, _, _, _) in enumerate(tqdm(loader)):
            label = torch.clamp(label, min=0.0, max=1.0)

            image = image.float().to(DEVICE)
            label = label.to(DEVICE)
            
            prediction = model(image)
            
            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > args.prediction_threshold).float()
            dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum())
            
            if epoch == 0:
                visualize.original_image(args, image, idx)
                visualize.original_image_with_label(args, image, label, idx)
            
            if epoch % int(args.epochs / 5) == 0 or args.epochs - 1 == epoch:
                visualize.original_image_with_prediction(args, image, prediction_binary, idx, epoch)
                img = visualize.original_image_with_prediction_color(args, image, prediction_binary, label, idx, epoch)
                if args.wandb:
                    img = wandb.Image(img, caption=f"Image {idx}")
                    wandb.log({
                        f'Validation result {idx}' : img,
                    }, step = epoch)
                

    dice = dice_score/len(loader)
    print(f"Dice score: {dice}")
    model.train()

    return model, dice

    
def train(
        args, DEVICE, model, loss_fn, optimizer, train_loader, val_loader
    ):
    count, pth_save_point = 0, 0
    best_loss = np.inf
    best_dice = -np.inf
    visualize.create_directories(args)
    
    if args.save_result:
        # mean losses
        losses = []
        b_loss = {'best_loss' : best_loss, 'epoch' : 0}
        # dice scores
        scores = []
        b_score = {'best_dice' : best_dice, 'epoch' : 0}
    
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
        
        if args.save_result:
            losses.append(mean_loss)
            scores.append(dice.item())

        print("Current loss: ", loss, mean_loss)
        if best_loss > mean_loss:
            print("=====New best model=====")
            best_loss, count = mean_loss, 0
            if args.save_result:
                b_loss['best_loss'] = best_loss
                b_loss['epoch'] = epoch
        else:
            count += 1
            if count > 5:
                args.dist_loss = True
                
        if best_dice < dice:
            best_dice = dice
            if args.save_result:
                b_score['best_dice'] = best_dice.item()
                b_score['epoch'] = epoch
        
        if args.wandb:
            log_results(
                mean_loss, dice, epoch
            )
            
        if epoch == args.epochs-1:
            if args.wandb:
                wandb.alert(f"Training Task Finished", f"Best dice score: {best_dice:.5f},  Best loss: {best_loss:.5f}")
            elif args.save_result:
                make_dir(args)
                # save loss_list
                with open(f'./result/{args.wandb_name}/loss/loss_list.pkl', 'wb') as f:
                    pickle.dump(losses, f)
                # save best_loss
                with open(f'./result/{args.wandb_name}/loss/best_loss.pkl', 'wb') as f:
                    pickle.dump(b_loss, f)
                # save score_list
                with open(f'./result/{args.wandb_name}/score/score_list.pkl', 'wb') as f:
                    pickle.dump(scores, f)                  
                # save best_score
                with open(f'./result/{args.wandb_name}/score/best_score.pkl', 'wb') as f:
                    pickle.dump(b_score, f)                  
                
                print(f"<<<<<Training Task {args.wandb_name} Finished>>>>>")