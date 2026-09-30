from config import config
from utils import set_seed
from utils import ClassicTraining, KFoldTraining, Predict_kfold
from omegaconf import DictConfig
import os
from ml_utils import ml_train
os.environ['WANDB_API_KEY'] = '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'

def run(config):
    set_seed(seed=config.seed)

    mode = config.mode

    if mode == 'classic_training':
        print('Classic training mode')
        ClassicTraining(config)
    elif mode == 'K_fold':
        print('K_fold training mode')
        KFoldTraining(config)
    elif mode == 'test_inference_K_fold':
        print('Test inference mode_K_fold')
        Predict_kfold(config)
    elif mode == "classic_ml":
        ml_train(config)
    else:
        raise ValueError(f'Invalid mode: {mode}')


if config.training.debug == True:
    print("Running in debug mode")
    config.training.number_of_debug_samples = 1000
    config.data.kfold.use_kfold = True

    run(config)
else:
    print("Running the full training")
    config.data.kfold.use_kfold = False
    config.training.debug = False
    run(config)