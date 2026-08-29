from dataclasses import dataclass


@dataclass
class Config:
    # -----------------------------
    # Dataset
    # -----------------------------

    dataset_dir: str = "data/raw/Flickr8k"
    features_dir: str = "data/processed/features"
    vocabulary_path: str = "artifacts/vocabulary.pkl"

    # -----------------------------
    # Model
    # -----------------------------

    feature_dim: int = 2048
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3

    # -----------------------------
    # Training
    # -----------------------------

    batch_size: int = 32
    num_epochs: int = 20

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # -----------------------------
    # Training utilities
    # -----------------------------

    patience: int = 3

    lr_factor: float = 0.5
    lr_patience: int = 1

    # -----------------------------
    # DataLoader
    # -----------------------------

    num_workers: int = 2

    # -----------------------------
    # Output
    # -----------------------------

    checkpoint_dir: str = "artifacts/checkpoints"
