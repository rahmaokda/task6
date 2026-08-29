import pickle

import torch

from src.models.caption_model import ImageCaptioningModel


def load_model(
    checkpoint_path="artifacts/checkpoints/best_model.pt",
    vocabulary_path="artifacts/vocabulary.pkl",
):
    """
    Load the trained captioning model and vocabulary.
    """

    device = torch.device("cpu")

    # -----------------------------------------
    # Load vocabulary
    # -----------------------------------------

    with open(
        vocabulary_path,
        "rb",
    ) as file:
        vocabulary = pickle.load(file)

    # -----------------------------------------
    # Load checkpoint
    # -----------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    # -----------------------------------------
    # Create model
    # -----------------------------------------

    model = ImageCaptioningModel(
        feature_dim=config.feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        vocab_size=len(vocabulary),
        pad_idx=vocabulary.word_to_id["<pad>"],
        dropout=config.dropout,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, vocabulary, device
