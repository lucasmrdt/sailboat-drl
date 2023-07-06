import wandb


def log(data={}, prefix=None):
    assert wandb.run is not None, 'wandb is not initialized'
    if prefix:
        data = {f'{prefix}/{k}': v for k, v in data.items()}
    wandb.log(data)
