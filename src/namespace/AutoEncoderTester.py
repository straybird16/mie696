import torch.nn as nn
ns = {
    "MODEL_NAME": ["AE"],
    #"MODEL_NAME": ["AE", "VAE", "SVAE"],
    "SPLIT": [0.9, 0.05, 0.05],
    #"IDC": [[i] for i in range(18)],
    "IDC": [None],
    "TRIM_DATA": [False],
    "FILTER_CORRCOEF": [True],
    "NORMALIZE_DATA": [True],
    "NORMALIZATION_SCHEME": ["standard_scaling"],
    "REMOVE_NOISE": [False],
    "NOISE_THRESHOLD": [i for i in range(1)],
    "NOISE_FACTOR": [1,],
    "NOISE_FRACTION": [0.5,],
    "PERPLEXITY": 10,
    "DISCRETIZE": [False],
    "INITIALIZATION": ["xavier_normal"],
    "ACTIVATION": ["leaky_relu"],
    "SIGMA": [1e-0],
    "LATENT_DIM": [16],
    "HIDDEN_DIM": [32],
    "GD_ALGORITHMS": ["SGD"],
    "WEIGHT_DECAY": [0e-6],
    "LEARNING_RATE": [1e-3],
    "GRAD_LIMIT": 1e2,
    "BATCH_SIZE": [1024],
    "EPOCHS": [1000],
    "ATTENTION": False,
    "ATT_E_DIM": 32,
    "usePEA": False, 
    "NUM_PROJECTION": 4,
    "THRESHOLD_UPPER": 0.95, 
    "THRESHOLD_LOWER": 0.00,
    "CRITERION": nn.MSELoss(reduction="sum"),
    "return_dict": {}, 
    "lcs_weights": None,
}