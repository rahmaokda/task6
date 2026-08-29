import torch

from src.models.caption_model import ImageCaptioningModel


def test_model_forward():

    batch_size = 4
    sequence_length = 10
    feature_dim = 2048
    embedding_dim = 256
    hidden_dim = 512
    vocab_size = 4722
    pad_idx = 0

    model = ImageCaptioningModel(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
    )

    image_features = torch.randn(
        batch_size,
        feature_dim,
    )

    captions = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )

    output = model(
        image_features,
        captions,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        vocab_size,
    )
