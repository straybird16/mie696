import torch.nn as nn
ns = {
    "MODEL_NAME": ["AE"],
    #"MODEL_NAME": ["AE", "VAE", "SVAE"],
    "SPLIT": [0.9, 0.05, 0.05],
    #"IDC": [[i] for i in range(18)],
    "IDC": [None],
    "TRIM_DATA": [False],
    "FILTER_CORRCOEF": [True],
    "REMOVE_NOISE": [False],
    "NOISE_THRESHOLD": [i for i in range(1)],
    "NOISE_FACTOR": [1,],
    "NOISE_FRACTION": [1,],
    "PERPLEXITY": 10,
    "DISCRETIZE": [True],
    "NORMALIZE_DATA": [True],
    "NORMALIZATION_SCHEME": ["standard_scaling"],
    "INITIALIZATION": ["xavier_normal"],
    "ACTIVATION": ["leaky_relu"],
    "SIGMA": [1e-0],
    "LATENT_DIM": [8],
    "HIDDEN_DIM": [32],
    "GD_ALGORITHMS": ["SGD"],
    "WEIGHT_DECAY": [0e-6],
    "LEARNING_RATE": [1e-4],
    "GRAD_LIMIT": 1e2,
    "BATCH_SIZE": [1024],
    "EPOCHS": [2000],
    "ATTENTION": False,
    "ATT_E_DIM": 32,
    "usePEA": False, 
    "NUM_PROJECTION": 4,
    "THRESHOLD_UPPER": 0.99, 
    "THRESHOLD_LOWER": 0.00,
    "CRITERION": nn.MSELoss(reduction="sum"),
    "return_dict": {}, 
    "lcs_weights": None,
}