import torch
import torch.nn as nn


class ImageCaptioningModel(nn.Module):
    """
    Image Captioning model using:

        ResNet50 features
            ↓
        Image projection
            ↓
        LSTM
            ↓
        Vocabulary prediction
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        pad_idx: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.image_projection = nn.Linear(
            feature_dim,
            embedding_dim,
        )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass during training.

        Args:
            image_features:
                [batch_size, 2048]

            captions:
                [batch_size, sequence_length]

        Returns:
            logits:
                [batch_size, sequence_length, vocab_size]
        """

        # Image features → embedding
        image_embedding = self.image_projection(
            image_features
        )

        image_embedding = self.dropout(
            image_embedding
        )

        # Caption tokens → word embeddings
        word_embeddings = self.embedding(
            captions
        )

        # Image is the first input to the LSTM
        image_embedding = image_embedding.unsqueeze(1)

        lstm_input = torch.cat(
            [
                image_embedding,
                word_embeddings,
            ],
            dim=1,
        )

        # LSTM
        lstm_output, _ = self.lstm(
            lstm_input
        )

        # Remove output corresponding to image
        lstm_output = lstm_output[:, 1:, :]

        # Predict vocabulary
        logits = self.output_layer(
            self.dropout(lstm_output)
        )

        return logits

    @torch.no_grad()
    def generate_caption(
        self,
        image_features: torch.Tensor,
        vocabulary,
        max_length: int = 30,
    ):
        """
        Generate a caption for a single image.
        """

        self.eval()

        device = next(self.parameters()).device

        # -----------------------------------------
        # Prepare image features
        # -----------------------------------------

        image_features = image_features.to(device)

        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        # -----------------------------------------
        # Project image into embedding space
        # -----------------------------------------

        image_embedding = self.image_projection(
            image_features
        )

        image_embedding = image_embedding.unsqueeze(1)

        # -----------------------------------------
        # Feed image into LSTM
        # -----------------------------------------

        _, hidden = self.lstm(
            image_embedding
        )

        # -----------------------------------------
        # Start caption
        # -----------------------------------------

        current_token = torch.tensor(
            [vocabulary.word_to_id["<start>"]],
            dtype=torch.long,
            device=device,
        )

        generated_tokens = []

        # -----------------------------------------
        # Generate words
        # -----------------------------------------

        for _ in range(max_length):

            word_embedding = self.embedding(
                current_token
            )

            word_embedding = word_embedding.unsqueeze(1)

            output, hidden = self.lstm(
                word_embedding,
                hidden,
            )

            logits = self.output_layer(
                output[:, -1, :]
            )

            next_token = logits.argmax(
                dim=-1
            )

            token_id = next_token.item()

            # Stop when <end> is generated
            if token_id == vocabulary.word_to_id["<end>"]:
                break

            # Don't add <start>
            if token_id != vocabulary.word_to_id["<start>"]:
                generated_tokens.append(token_id)

            current_token = next_token

        # Convert token IDs → words
        caption = vocabulary.decode(
            generated_tokens
        )

        return caption
