from omegaconf import OmegaConf


config = OmegaConf.create({
    "seed": 42,
    "mode": "classic_training",
    "device": "cuda",
    "in_features": 288,
    "training": {
        "debug": False,
        "number_of_debug_samples": 1000,
        "lr": 1e-3,
        "epochs": 60,
        "batch_size": 32,
    },
    "data": {
        "kfold": {
            "use_kfold": True,
            "n_splits": 6,        
            "val_size": 0.2,
        }
    },
    "paths": {
        "checkpoint_dir": "./checkpoints",
        "train_csv": "./data/train.csv",
    },
    "logging": {
        "wandb_project_name": "titanic-classification"
    }
})