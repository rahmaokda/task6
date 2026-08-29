import pickle

from src.data.preprocessing import Flickr8kPreprocessor
from src.data.dataset import Flickr8kDataset, caption_collate_fn


def test_dataset():

    preprocessor = Flickr8kPreprocessor()

    (
        image_captions,
        train_images,
        val_images,
        test_images,
    ) = preprocessor.prepare()

    with open(
        "artifacts/vocabulary.pkl",
        "rb",
    ) as file:
        vocabulary = pickle.load(file)

    dataset = Flickr8kDataset(
        image_names=train_images,
        image_captions=image_captions,
        features_dir="data/processed/features",
        vocabulary=vocabulary,
    )

    # 5 captions for every training image
    assert len(dataset) == len(train_images) * 5

    image_feature, caption = dataset[0]

    assert image_feature.shape == (2048,)
    assert caption.ndim == 1
    assert caption[0].item() == vocabulary.word_to_id["<start>"]
    assert caption[-1].item() == vocabulary.word_to_id["<end>"]


def test_collate_fn():

    preprocessor = Flickr8kPreprocessor()

    (
        image_captions,
        train_images,
        _,
        _,
    ) = preprocessor.prepare()

    with open(
        "artifacts/vocabulary.pkl",
        "rb",
    ) as file:
        vocabulary = pickle.load(file)

    dataset = Flickr8kDataset(
        image_names=train_images[:2],
        image_captions=image_captions,
        features_dir="data/processed/features",
        vocabulary=vocabulary,
    )

    batch = [
        dataset[0],
        dataset[1],
        dataset[2],
    ]

    image_features, captions = caption_collate_fn(batch)

    assert image_features.shape == (3, 2048)
    assert captions.shape[0] == 3

    # All captions in the batch have equal length
    assert captions.ndim == 2
