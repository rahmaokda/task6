import pickle

import torch
import torch.nn as nn

from src.data.dataset import (
    Flickr8kDataset,
    caption_collate_fn,
)
from src.data.preprocessing import Flickr8kPreprocessor
from src.models.caption_model import ImageCaptioningModel


def test_training_batch():

    # -----------------------------------------
    # Prepare dataset
    # -----------------------------------------

    preprocessor = Flickr8kPreprocessor()

    (
        image_captions,
        train_images,
        _,
        _,
    ) = preprocessor.prepare()

    # -----------------------------------------
    # Load vocabulary
    # -----------------------------------------

    with open(
        "artifacts/vocabulary.pkl",
        "rb",
    ) as file:
        vocabulary = pickle.load(file)

    # -----------------------------------------
    # Create a small dataset
    # -----------------------------------------

    dataset = Flickr8kDataset(
        image_names=train_images[:10],
        image_captions=image_captions,
        features_dir="data/processed/features",
        vocabulary=vocabulary,
    )

    # -----------------------------------------
    # Create a small batch
    # -----------------------------------------

    batch = [
        dataset[i]
        for i in range(8)
    ]

    image_features, captions = caption_collate_fn(
        batch
    )

    # -----------------------------------------
    # Create model
    # -----------------------------------------

    model = ImageCaptioningModel(
        feature_dim=2048,
        embedding_dim=256,
        hidden_dim=512,
        vocab_size=len(vocabulary),
        pad_idx=vocabulary.word_to_id["<pad>"],
    )

    # -----------------------------------------
    # Teacher forcing
    # -----------------------------------------

    inputs = captions[:, :-1]
    targets = captions[:, 1:]

    # -----------------------------------------
    # Forward pass
    # -----------------------------------------

    logits = model(
        image_features,
        inputs,
    )

    # -----------------------------------------
    # Check dimensions
    # -----------------------------------------

    assert logits.shape[0] == 8
    assert logits.shape[1] == inputs.shape[1]
    assert logits.shape[2] == len(vocabulary)

    # -----------------------------------------
    # Loss
    # -----------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.word_to_id["<pad>"]
    )

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    # Loss should be a valid scalar
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # -----------------------------------------
    # Backward pass
    # -----------------------------------------

    loss.backward()

    # At least one parameter should have gradients.
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert len(gradients) > 0

    for gradient in gradients:
        assert torch.isfinite(gradient).all()
