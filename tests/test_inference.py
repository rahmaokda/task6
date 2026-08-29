import torch

from src.inference.predict import load_model


def test_caption_generation():

    model, vocabulary, device = load_model()

    feature_path = (
        "data/processed/features/"
        "1000268201_693b08cb0e.pt"
    )

    features = torch.load(
        feature_path,
        map_location=device,
    )

    caption = model.generate_caption(
        features,
        vocabulary,
    )

    print("\nGenerated caption:", caption)

    assert isinstance(caption, str)

    assert len(caption) > 0
