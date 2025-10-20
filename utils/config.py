SYSTEM_CONFIG = {
    'N': 20,
    'K': 4,
    'M': 5,
    'tau': 1.0,
    'communication_rounds': 500,
    'device': 'cuda:0'
}

VEHICLE_DYNAMICS_CONFIG = {
    'a_max': 0.73,
    'b_max': 1.67,
    'v_des': 30.0,
    'delta': 4,
    'd_min': 2.0,
    't_min': 1.5,
    'vehicle_length': 5.0
}

CHANNEL_CONFIG = {
    'B': 56e6,
    'N0': 1e-9,
    'alpha': 3.76,
    'P_error': 0.1,
    'f_c': 2.4e9
}

RESOURCE_ALLOCATION_CONFIG = {
    'D_n': 10.08e6,
    'B': 56e6,
    'e_max': 0.1,
    'mu': 1e7,
    'kappa': 1e-28,
    'G_n': 0.5e9,
    'P_n': 15
}

FLMD_CONFIG = {
    'lambda_theta': 0.1,
    'beta': 1.0,
    'mu_over_L': 0.01
}

FEDERATED_LEARNING_CONFIG = {
    'learning_rate': 1e-4,
    'local_epochs': 1,
    'batch_size': 32,
    'gradient_compression_ratio': 0.3
}

MAPPO_CONFIG = {
    'learning_rate': 1e-4,
    'gamma': 0.98,
    'lambda_gae': 0.95,
    'epsilon_clip': 0.2,
    'num_epochs': 3,
    'hidden_dim': 128
}

TSFEN_CONFIG = {
    'd_model': 64,
    'num_heads': 8,
    'hidden_size': 128,
    'num_agents': 20,
    'temperature': 0.08
}

DEEPLABV3PLUS_CONFIG = {
    'num_classes': 4,
    'output_stride': 16,
    'pretrained': True
}

DATASET_CONFIG = {
    'dataset_name': 'AI4MARS',
    'num_classes': 4,
    'input_size': (513, 513),
    'alpha': 0.5,
    'train_samples': 10500,
    'train_labels': 98000
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 500,
    'validation_split': 0.2,
    'early_stopping_patience': 20
}

OPTIMIZATION_CONFIG = {
    'optimizer': 'AdamW',
    'weight_decay': 1e-5,
    'gradient_clip': 1.0
}

LOGGING_CONFIG = {
    'log_interval': 10,
    'save_interval': 50,
    'checkpoint_dir': './checkpoints'
}

