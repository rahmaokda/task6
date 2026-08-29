from src.data.vocabulary import Vocabulary


def test_special_tokens():

    vocab = Vocabulary()

    assert vocab.word_to_id["<pad>"] == 0
    assert vocab.word_to_id["<unk>"] == 1
    assert vocab.word_to_id["<start>"] == 2
    assert vocab.word_to_id["<end>"] == 3


def test_build_vocabulary():

    captions = [
        "A dog is running.",
        "A dog is playing.",
        "A cat is sleeping.",
    ]

    vocab = Vocabulary(min_frequency=2)

    vocab.build(captions)

    assert "dog" in vocab.word_to_id
    assert "is" in vocab.word_to_id

    # "cat" appears only once.
    assert "cat" not in vocab.word_to_id


def test_numericalize():

    captions = [
        "A dog is running.",
        "A dog is playing.",
    ]

    vocab = Vocabulary(min_frequency=1)
    vocab.build(captions)

    sequence = vocab.numericalize(
        "A dog is running."
    )

    assert sequence[0] == vocab.word_to_id["<start>"]
    assert sequence[-1] == vocab.word_to_id["<end>"]


def test_decode():

    captions = [
        "A dog is running.",
    ]

    vocab = Vocabulary(min_frequency=1)
    vocab.build(captions)

    sequence = vocab.numericalize(
        "A dog is running."
    )

    decoded = vocab.decode(sequence)

    assert decoded == "a dog is running"
