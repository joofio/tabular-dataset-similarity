"""Perturbation configuration in Python form."""

PERTURBATIONS = [
    {"name": "mean_variance_drift", "params": {"mean_shift": 0.1, "scale": 1.1}},
    {"name": "noise_injection", "params": {"std": 0.05}},
    {"name": "category_flip", "params": {"flip_prob": 0.1}},
    {"name": "correlation_mix", "params": {"alpha": 0.5}},
    {"name": "missingness_mcar", "params": {"rate": 0.1, "mechanism": "mcar"}},
]
