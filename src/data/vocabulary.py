from collections import Counter


class Vocabulary:
    """
    Maps words to integer IDs and integer IDs back to words.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"

    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency

        # Special tokens
        self.word_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3,
        }

        self.id_to_word = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
            2: self.START_TOKEN,
            3: self.END_TOKEN,
        }

    def __len__(self):
        return len(self.word_to_id)

    def tokenize(self, caption: str) -> list[str]:
        """
        Convert a caption into normalized word tokens.
        """

        caption = caption.lower().strip()

        # Keep only alphabetic words and simple apostrophes.
        tokens = []

        for word in caption.split():
            word = word.strip(".,!?;:\"'()[]{}")

            if word:
                tokens.append(word)

        return tokens

    def build(self, captions: list[str]) -> None:
        """
        Build the vocabulary from a collection of captions.
        """

        counter = Counter()

        for caption in captions:
            tokens = self.tokenize(caption)
            counter.update(tokens)

        # Sort to make vocabulary construction deterministic.
        words = sorted(
            word
            for word, count in counter.items()
            if count >= self.min_frequency
        )

        for word in words:
            if word not in self.word_to_id:
                index = len(self.word_to_id)

                self.word_to_id[word] = index
                self.id_to_word[index] = word

    def numericalize(self, caption: str) -> list[int]:
        """
        Convert a caption into a sequence of integer IDs.
        """

        tokens = self.tokenize(caption)

        tokens = (
            [self.START_TOKEN]
            + tokens
            + [self.END_TOKEN]
        )

        return [
            self.word_to_id.get(
                token,
                self.word_to_id[self.UNK_TOKEN],
            )
            for token in tokens
        ]

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back into a human-readable caption.
        """

        words = []

        for token_id in token_ids:

            word = self.id_to_word.get(
                token_id,
                self.UNK_TOKEN,
            )

            if word == self.START_TOKEN:
                continue

            if word == self.END_TOKEN:
                break

            if word == self.PAD_TOKEN:
                continue

            words.append(word)

        return " ".join(words)
