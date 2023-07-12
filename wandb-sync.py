import wandb

if __name__ == '__main__':
    wandb.tensorboard.patch(root_logdir='logs/max-vmc-v6')
    wandb.init(project='sailboat-drl', sync_tensorboard=True)
    wandb.finish()