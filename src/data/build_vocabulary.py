import pickle

from src.data.preprocessing import Flickr8kPreprocessor
from src.data.vocabulary import Vocabulary


def main():

    # --------------------------------
    # 1. Prepare Flickr8k
    # --------------------------------

    preprocessor = Flickr8kPreprocessor()

    (
        image_captions,
        train_images,
        val_images,
        test_images,
    ) = preprocessor.prepare()

    # --------------------------------
    # 2. Get TRAINING captions only
    # --------------------------------

    train_captions = preprocessor.get_captions_for_images(
        image_captions,
        train_images,
    )

    print("Training images:", len(train_images))
    print("Validation images:", len(val_images))
    print("Test images:", len(test_images))

    print("Training captions:", len(train_captions))

    # --------------------------------
    # 3. Build vocabulary
    # --------------------------------

    vocabulary = Vocabulary(
        min_frequency=2
    )

    vocabulary.build(train_captions)

    print("Vocabulary size:", len(vocabulary))

    # --------------------------------
    # 4. Save vocabulary
    # --------------------------------

    output_path = "artifacts/vocabulary.pkl"

    with open(output_path, "wb") as file:
        pickle.dump(vocabulary, file)

    print(f"Vocabulary saved to: {output_path}")


if __name__ == "__main__":
    main()
