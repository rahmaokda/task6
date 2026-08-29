from src.data.preprocessing import Flickr8kPreprocessor


def test_flickr8k_preparation():

    preprocessor = Flickr8kPreprocessor()

    (
        image_captions,
        train_images,
        val_images,
        test_images,
    ) = preprocessor.prepare()

    # Dataset size
    assert len(image_captions) == 8091

    # Every image has five captions
    for captions in image_captions.values():
        assert len(captions) == 5

    # No data leakage
    assert set(train_images).isdisjoint(val_images)
    assert set(train_images).isdisjoint(test_images)
    assert set(val_images).isdisjoint(test_images)

    # All images are accounted for
    total_images = (
        len(train_images)
        + len(val_images)
        + len(test_images)
    )

    assert total_images == 8091
