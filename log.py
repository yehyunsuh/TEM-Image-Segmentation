import wandb

def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )


def log_results(loss, dice):
    wandb.log({
        'Train Loss': loss,
        # 'Dice Score': dice,
    })