# config.py

# Base configuration for shared hyperparameters
CONFIG = {
    "epochs": 100,
    "batch_size": 8,
    "test_batch_size": 16,
    "num_workers": 4,
    "eval_interval": 1,
    
    "ignore_label": 255,
    "n_times": 12,
    "model_name": "MTCNet-Base",
    "use_dp": False,
    
    "optimizer_name": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    
    "scheduler_name": "warmuppolylr",
    "scheduler_power": 0.9,
    "warmup_iters": 10,
    "warmup_ratio": 0.1,
    "seed": 3407
}

# Dataset-specific configurations
DATASET_CONFIG = {
    "M2Change-CZ1": {
        "dataset_root": " ",
        "log_dir": " ",
        "val_model_path": " ",
        "val_save_dir": " "
    },
    "M2Change-CZ2": {
        "dataset_root": " ",
        "log_dir": " ",
        "val_model_path": " ",
        "val_save_dir": " "
    }
}