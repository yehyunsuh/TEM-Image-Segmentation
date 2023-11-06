import wandb

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