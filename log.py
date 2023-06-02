import wandb

def initiate_wandb(args):
    if args.wandb:
        if args.wandb_sweep:
            wandb.init(
                project=f"{args.wandb_project}", 
                entity=f"{args.wandb_entity}",
            )
            args.batch_size        = int(wandb.config['batch_size'])
            args.learning_rate     = int(wandb.config['learning_rate'])
            args.seed              = int(wandb.config['seed'])
        else:
            wandb.init(
                project=f"{args.wandb_project}", 
                entity=f"{args.wandb_entity}",
                name=f"{args.wandb_name}"
            )