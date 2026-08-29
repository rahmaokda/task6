import statistics
from pathlib import Path

import pandas as pd
import torch

from src.inference.predict import load_model
from src.evaluation.metrics import calculate_all_metrics


CAPTIONS_PATH = "data/raw/Flickr8k/captions.txt"
FEATURES_DIR = Path("data/processed/features")


def load_test_images():
    """
    Recreate the same Flickr8k train/validation/test split
    used during training.
    """

    from src.data.preprocessing import Flickr8kPreprocessor

    preprocessor = Flickr8kPreprocessor()

    image_captions, train_images, val_images, test_images = (
        preprocessor.prepare()
    )

    return image_captions, test_images


def main():

    print("Loading model...")

    model, vocabulary, device = load_model()

    print("Model loaded.")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Device: {device}")

    # --------------------------------------------------
    # Load test images
    # --------------------------------------------------

    image_captions, test_images = load_test_images()

    print(f"Test images: {len(test_images)}")

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    all_scores = []

    for index, image_name in enumerate(test_images, start=1):

        feature_path = FEATURES_DIR / f"{image_name.rsplit('.', 1)[0]}.pt"

        if not feature_path.exists():
            print(f"Skipping missing feature: {feature_path}")
            continue

        # Load pre-extracted ResNet feature
        features = torch.load(
            feature_path,
            map_location=device,
            weights_only=True,
        )

        # Generate caption
        hypothesis = model.generate_caption(
            features,
            vocabulary,
            max_length=30,
        )

        # Human reference captions
        references = image_captions[image_name]

        # Calculate metrics
        scores = calculate_all_metrics(
            references,
            hypothesis,
        )

        all_scores.append(scores)

        # Print progress
        if index % 50 == 0 or index == 1:
            print(
                f"[{index}/{len(test_images)}] "
                f"{image_name}"
            )
            print(f"  Generated: {hypothesis}")

    # --------------------------------------------------
    # Aggregate results
    # --------------------------------------------------

    if not all_scores:
        raise RuntimeError("No test images were evaluated.")

    metric_names = all_scores[0].keys()

    average_scores = {}

    for metric in metric_names:
        average_scores[metric] = statistics.mean(
            score[metric]
            for score in all_scores
        )

    # --------------------------------------------------
    # Print final results
    # --------------------------------------------------

    print()
    print("=" * 50)
    print("FINAL TEST SET RESULTS")
    print("=" * 50)

    print(f"Images evaluated: {len(all_scores)}")
    print()

    for metric, score in average_scores.items():
        print(f"{metric:10s}: {score:.4f}")

    print("=" * 50)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = Path("artifacts/test_results.csv")

    df = pd.DataFrame(
        [average_scores]
    )

    df.insert(
        0,
        "images_evaluated",
        len(all_scores),
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
