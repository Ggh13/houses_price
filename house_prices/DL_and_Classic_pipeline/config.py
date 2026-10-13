from omegaconf import OmegaConf


config = OmegaConf.create({
    "seed": 42,
    "mode": "stacking_ml",
    "type_classic": "RandomForest",
    "device": "cuda",
    "training": {
        "debug": False,
        "number_of_debug_samples": 1000,
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 64,
        "weight_decay": 0.0,
        "max_patience": 15,
    },
    "model": {
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.05,
    },
    "scheduler": {
        "patience": 5,
        "factor": 0.1,
    },
    "data": {
        "kfold": {
            "n_splits": 5,
            "val_size": 0.2,
        },
        "shuffle": True,
    },
    "paths": {
        "checkpoint_dir": "./checkpoints",
        "train_csv": "./data/train.csv",
        "test_csv": "./data/test.csv",
    },
    "logging": {
        "wandb_project_name": "titanic-classification"
    },
    "files": {
        "data": "./data/",
        "train": "train.csv"
    },
})
