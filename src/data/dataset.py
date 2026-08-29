from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.vocabulary import Vocabulary


class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k.

    Each item contains:
        - a pre-extracted image feature vector
        - a tokenized caption
    """

    def __init__(
        self,
        image_names: list[str],
        image_captions: dict[str, list[str]],
        features_dir: str,
        vocabulary: Vocabulary,
    ):
        self.image_names = image_names
        self.image_captions = image_captions
        self.features_dir = Path(features_dir)
        self.vocabulary = vocabulary

        # Create one sample for every image-caption pair.
        self.samples = []

        for image_name in self.image_names:

            captions = self.image_captions[image_name]

            for caption in captions:

                self.samples.append(
                    (image_name, caption)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_name, caption = self.samples[index]

        # --------------------------------
        # Load pre-extracted image feature
        # --------------------------------

        feature_path = (
            self.features_dir
            / f"{Path(image_name).stem}.pt"
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        image_feature = torch.load(
            feature_path,
            map_location="cpu",
        )

        # --------------------------------
        # Convert caption → token IDs
        # --------------------------------

        caption_ids = self.vocabulary.numericalize(
            caption
        )

        caption_ids = torch.tensor(
            caption_ids,
            dtype=torch.long,
        )

        return image_feature, caption_ids


def caption_collate_fn(batch):
    """
    Collate function for variable-length captions.
    """

    image_features, captions = zip(*batch)

    image_features = torch.stack(image_features)

    max_length = max(
        len(caption)
        for caption in captions
    )

    pad_id = 0

    padded_captions = torch.full(
        (len(captions), max_length),
        fill_value=pad_id,
        dtype=torch.long,
    )

    for i, caption in enumerate(captions):
        padded_captions[
            i,
            :len(caption)
        ] = caption

    return image_features, padded_captions
