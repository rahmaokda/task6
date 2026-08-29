import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.dataset import (
    Flickr8kDataset,
    caption_collate_fn,
)
from src.data.preprocessing import Flickr8kPreprocessor
from src.models.caption_model import ImageCaptioningModel


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0

    for image_features, captions in dataloader:

        image_features = image_features.to(device)
        captions = captions.to(device)

        # -----------------------------------------
        # Teacher forcing
        # -----------------------------------------

        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        # -----------------------------------------
        # Forward pass
        # -----------------------------------------

        logits = model(
            image_features,
            inputs,
        )

        # -----------------------------------------
        # Reshape for CrossEntropyLoss
        # -----------------------------------------

        logits = logits.reshape(
            -1,
            logits.size(-1),
        )

        targets = targets.reshape(-1)

        # -----------------------------------------
        # Loss
        # -----------------------------------------

        loss = criterion(
            logits,
            targets,
        )

        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate the model on the validation set.
    """

    model.eval()

    total_loss = 0.0

    for image_features, captions in dataloader:

        image_features = image_features.to(device)
        captions = captions.to(device)

        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        logits = model(
            image_features,
            inputs,
        )

        logits = logits.reshape(
            -1,
            logits.size(-1),
        )

        targets = targets.reshape(-1)

        loss = criterion(
            logits,
            targets,
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():

    config = Config()

    # -----------------------------------------
    # Device
    # -----------------------------------------

    device = torch.device("cpu")

    print(f"Using device: {device}")

    # -----------------------------------------
    # Prepare dataset
    # -----------------------------------------

    preprocessor = Flickr8kPreprocessor(
        dataset_dir=config.dataset_dir
    )

    (
        image_captions,
        train_images,
        val_images,
        test_images,
    ) = preprocessor.prepare()

    # -----------------------------------------
    # Load vocabulary
    # -----------------------------------------

    with open(
        config.vocabulary_path,
        "rb",
    ) as file:
        vocabulary = pickle.load(file)

    print(
        f"Vocabulary size: {len(vocabulary)}"
    )

    # -----------------------------------------
    # Create datasets
    # -----------------------------------------

    train_dataset = Flickr8kDataset(
        image_names=train_images,
        image_captions=image_captions,
        features_dir=config.features_dir,
        vocabulary=vocabulary,
    )

    val_dataset = Flickr8kDataset(
        image_names=val_images,
        image_captions=image_captions,
        features_dir=config.features_dir,
        vocabulary=vocabulary,
    )

    # -----------------------------------------
    # Create dataloaders
    # -----------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=caption_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=caption_collate_fn,
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # -----------------------------------------
    # Create model
    # -----------------------------------------

    model = ImageCaptioningModel(
        feature_dim=config.feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        vocab_size=len(vocabulary),
        pad_idx=vocabulary.word_to_id["<pad>"],
        dropout=config.dropout,
    )

    model.to(device)

    # -----------------------------------------
    # Loss
    # -----------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.word_to_id["<pad>"]
    )

    # -----------------------------------------
    # Optimizer
    # -----------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # -----------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    # -----------------------------------------
    # Checkpoint directory
    # -----------------------------------------

    checkpoint_dir = Path(
        config.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    # -----------------------------------------
    # Training loop
    # -----------------------------------------

    for epoch in range(config.num_epochs):

        print(
            f"\nEpoch {epoch + 1}/{config.num_epochs}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.6f}"
        )

        # -----------------------------------------
        # Save best model
        # -----------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            checkpoint_path = (
                checkpoint_dir
                / "best_model.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                    "config": config,
                },
                checkpoint_path,
            )

            print(
                f"Saved best model → {checkpoint_path}"
            )

        else:

            epochs_without_improvement += 1

        # -----------------------------------------
        # Early stopping
        # -----------------------------------------

        if (
            epochs_without_improvement
            >= config.patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
