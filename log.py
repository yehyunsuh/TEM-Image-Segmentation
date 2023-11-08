import wandb
import os

def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )
        wandb.config.update(args)


def log_results(loss, dice, epoch):
    wandb.log({
        'Train Loss': loss,
        'Dice Score': dice
    }, step = epoch)
   

def make_dir(args):
    if not os.path.exists(f'./result'):
        os.mkdir(f'./result')
    if not os.path.exists(f'./result/{args.wandb_name}'):
        os.mkdir(f'./result/{args.wandb_name}')
    if not os.path.exists(f'./result/{args.wandb_name}/loss'):
        os.mkdir(f'./result/{args.wandb_name}/loss')
    if not os.path.exists(f'./result/{args.wandb_name}/score'):
        os.mkdir(f'./result/{args.wandb_name}/score')
    