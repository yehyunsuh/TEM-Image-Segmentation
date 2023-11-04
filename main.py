import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import preprocess

from dataset import load_data
from log import initiate_wandb
from model import get_model, get_pretrained_model
from train import train

def main(args):
    preprocess.customize_seed(args.seed) # set seed
    initiate_wandb(args) # initiate wandb
    
    train_loader, val_loader = load_data(args)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Torch is running on {DEVICE}')
    
    if args.pretrained:
        model = get_pretrained_model(args, DEVICE)
    else:
        model = get_model(args, DEVICE)
    
    if args.weight_on_border:
        print("***** use weight loss *****")
        if args.s_weight:
            print("------ use special weight ------")
    else:
        print("***** use ordinary loss *****")
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.loss_weight], device=DEVICE))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, DEVICE, model, loss_fn, optimizer, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained model or not')
    
    ## settings
    parser.add_argument('--seed', type=int, default=2023, help='seed customization for result reproduction')
    
    ## hyperparameters - data
    parser.add_argument('--data_dir', type=str, default='./dataset', help='data directory')
    parser.add_argument('--train', action='store_true', help='train data or validation data')
    parser.add_argument('--pore', action='store_true', help='pore or nano_particle')
    parser.add_argument('--image_resize', type=int, default=512, help='image resize value')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--augmentation', action='store_true', help='whether to use augmentation')
    parser.add_argument('--prediction_threshold', type=float, default=0.5, help='threshold that makes prediction binary')
    
    # distribution loss
    parser.add_argument('--max_pore', type=int, default=155, help='maximum number of pore in an image')
    parser.add_argument('--distribution_loss', action='store_true', help='whether to use loss based on distribution')
    parser.add_argument('--dist_loss', action='store_true', help='become True when count becomes 5 and use distribution loss')
    parser.add_argument('--weight_num', type=int, default=1e-4, help='weight on the number of pores (or np)')
    parser.add_argument('--weight_pixel', type=int, default=1e-4, help='weight on the number of pixels of pores (or np)')
    
    # weight on border
    parser.add_argument('--weight_on_border', action='store_true', help='whether to wight border') # whether to give weight on border 
    parser.add_argument('--weight', type=int, default=40, help='how much weight to impose on border') # how much weight to give on border
    parser.add_argument('--s_weight', action='store_true', help='use special weight') # give weight by method Yehyun suggested
    
    ## hyperparameters - model
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--loss_weight', type=int, default=1, help='number of epochs')
    
    ## wandb
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb or not')
    parser.add_argument('--wandb_project', type=str, default="TEM_Image_Segmentation", help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="tem_seg", help='wandb entity name')
    parser.add_argument('--wandb_name', type=str, default="JY_exp_bweight_40", help='wandb name')
    
    args = parser.parse_args()
    main(args)