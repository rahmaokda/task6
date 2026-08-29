from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def calculate_bleu(
    references: list[str],
    hypothesis: str,
) -> dict[str, float]:
    """
    Calculate BLEU-1 through BLEU-4.

    Args:
        references:
            List of human-written reference captions.

        hypothesis:
            Model-generated caption.

    Returns:
        Dictionary containing BLEU-1, BLEU-2,
        BLEU-3, and BLEU-4 scores.
    """

    reference_tokens = [
        reference.lower().split()
        for reference in references
    ]

    hypothesis_tokens = hypothesis.lower().split()

    smoothing = SmoothingFunction().method1

    scores = {}

    scores["BLEU-1"] = sentence_bleu(
        reference_tokens,
        hypothesis_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing,
    )

    scores["BLEU-2"] = sentence_bleu(
        reference_tokens,
        hypothesis_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing,
    )

    scores["BLEU-3"] = sentence_bleu(
        reference_tokens,
        hypothesis_tokens,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smoothing,
    )

    scores["BLEU-4"] = sentence_bleu(
        reference_tokens,
        hypothesis_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    return scores


def calculate_rouge(
    references: list[str],
    hypothesis: str,
) -> float:
    """
    Calculate ROUGE-L using the best matching
    reference caption.
    """

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    scores = []

    for reference in references:
        result = scorer.score(
            reference,
            hypothesis,
        )

        scores.append(
            result["rougeL"].fmeasure
        )

    return max(scores)


def calculate_meteor(
    references: list[str],
    hypothesis: str,
) -> float:
    """
    Calculate METEOR using the best matching
    reference caption.
    """

    hypothesis_tokens = hypothesis.lower().split()

    reference_tokens = [
        reference.lower().split()
        for reference in references
    ]

    scores = []

    for reference in reference_tokens:
        scores.append(
            meteor_score(
                [reference],
                hypothesis_tokens,
            )
        )

    return max(scores)


def calculate_all_metrics(
    references: list[str],
    hypothesis: str,
) -> dict[str, float]:
    """
    Calculate all evaluation metrics.
    """

    results = {}

    results.update(
        calculate_bleu(
            references,
            hypothesis,
        )
    )

    results["ROUGE-L"] = calculate_rouge(
        references,
        hypothesis,
    )

    results["METEOR"] = calculate_meteor(
        references,
        hypothesis,
    )

    return results
