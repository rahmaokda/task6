import argparse
import pickle

import torch

from src.models.caption_model import ImageCaptioningModel
from src.vision.feature_extractor import ResNetFeatureExtractor


def load_model(
    checkpoint_path="artifacts/checkpoints/best_model.pt",
    vocabulary_path="artifacts/vocabulary.pkl",
):
    """
    Load the trained captioning model and vocabulary.
    """

    device = torch.device("cpu")

    # Load vocabulary
    with open(
        vocabulary_path,
        "rb",
    ) as file:
        vocabulary = pickle.load(file)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    # Create model
    model = ImageCaptioningModel(
        feature_dim=config.feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        vocab_size=len(vocabulary),
        pad_idx=vocabulary.word_to_id["<pad>"],
        dropout=config.dropout,
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, vocabulary, device


def generate_caption(
    image_path,
    model,
    vocabulary,
    device,
    max_length=30,
):
    """
    Generate a caption for an image.

    Args:
        image_path:
            Path to the input image.

        model:
            Trained image captioning model.

        vocabulary:
            Vocabulary used during training.

        device:
            Device used for inference.

        max_length:
            Maximum number of tokens to generate.

    Returns:
        Generated caption as a string.
    """

    # Create feature extractor
    extractor = ResNetFeatureExtractor(
        device=str(device)
    )

    # Extract 2048-dimensional image features
    image_features = extractor.extract(
        image_path
    )

    # Generate caption
    #
    # IMPORTANT:
    # model.generate_caption() already converts
    # token IDs into a string using vocabulary.decode().
    caption = model.generate_caption(
        image_features,
        vocabulary,
        max_length=max_length,
    )

    return caption


def main():
    """
    Command-line interface for image caption generation.
    """

    parser = argparse.ArgumentParser(
        description="Generate a caption for an image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum caption length.",
    )

    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/best_model.pt",
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--vocabulary",
        default="artifacts/vocabulary.pkl",
        help="Path to the vocabulary file.",
    )

    args = parser.parse_args()

    # Load trained model
    model, vocabulary, device = load_model(
        checkpoint_path=args.checkpoint,
        vocabulary_path=args.vocabulary,
    )

    # Generate caption
    caption = generate_caption(
        image_path=args.image,
        model=model,
        vocabulary=vocabulary,
        device=device,
        max_length=args.max_length,
    )

    print()
    print("Image:", args.image)
    print("Caption:", caption)
    print()


if __name__ == "__main__":
    main()
