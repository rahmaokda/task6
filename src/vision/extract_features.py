from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.vision.feature_extractor import ResNetFeatureExtractor


def extract_all_features(
    images_dir: str,
    output_dir: str,
    device: str = "cpu",
):
    """
    Extract and cache ResNet50 features for all images.

    Existing feature files are skipped so the process
    can safely resume after interruption.
    """

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Supported image formats
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    )

    print(f"Found {len(image_paths)} images.")

    extractor = ResNetFeatureExtractor(
        device=device
    )

    processed = 0
    skipped = 0
    failed = 0

    for image_path in tqdm(
        image_paths,
        desc="Extracting image features",
    ):

        output_path = (
            output_dir
            / f"{image_path.stem}.pt"
        )

        # Skip already processed images
        if output_path.exists():
            skipped += 1
            continue

        try:
            features = extractor.extract(
                image_path
            )

            torch.save(
                features,
                output_path,
            )

            processed += 1

        except Exception as error:
            failed += 1

            print(
                f"\nFailed: {image_path.name}"
            )
            print(f"Error: {error}")

    print("\nFeature extraction complete.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":

    extract_all_features(
        images_dir="data/raw/Flickr8k/Images",
        output_dir="data/processed/features",
        device="cpu",
    )
