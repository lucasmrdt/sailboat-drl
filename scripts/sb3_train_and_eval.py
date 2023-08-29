import allow_local_package_imports

from sb3_train import train_model
from sb3_eval import eval_model

if __name__ == '__main__':
    train_model()
    eval_model()
