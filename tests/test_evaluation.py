from src.evaluation.metrics import (
    calculate_bleu,
    calculate_rouge,
    calculate_meteor,
    calculate_all_metrics,
)


def test_bleu():

    references = [
        "a dog is running in the park",
        "a dog runs through the park",
        "a dog is playing outside",
    ]

    hypothesis = "a dog is running in the park"

    scores = calculate_bleu(
        references,
        hypothesis,
    )

    assert "BLEU-1" in scores
    assert "BLEU-2" in scores
    assert "BLEU-3" in scores
    assert "BLEU-4" in scores

    assert 0 <= scores["BLEU-1"] <= 1
    assert 0 <= scores["BLEU-2"] <= 1
    assert 0 <= scores["BLEU-3"] <= 1
    assert 0 <= scores["BLEU-4"] <= 1


def test_rouge():

    references = [
        "a dog is running in the park",
        "a dog runs through the park",
    ]

    hypothesis = "a dog is running in the park"

    score = calculate_rouge(
        references,
        hypothesis,
    )

    assert 0 <= score <= 1


def test_meteor():

    references = [
        "a dog is running in the park",
        "a dog runs through the park",
    ]

    hypothesis = "a dog is running in the park"

    score = calculate_meteor(
        references,
        hypothesis,
    )

    assert 0 <= score <= 1


def test_all_metrics():

    references = [
        "a dog is running in the park",
        "a dog runs through the park",
    ]

    hypothesis = "a dog is running in the park"

    scores = calculate_all_metrics(
        references,
        hypothesis,
    )

    assert len(scores) == 6

    assert "BLEU-1" in scores
    assert "BLEU-2" in scores
    assert "BLEU-3" in scores
    assert "BLEU-4" in scores
    assert "ROUGE-L" in scores
    assert "METEOR" in scores
