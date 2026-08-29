from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNetFeatureExtractor:
    """
    Extracts 2048-dimensional image features using
    a pretrained ResNet50 model.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT

        model = models.resnet50(weights=weights)

        # Remove the final classification layer.
        # ResNet50 normally outputs 1000 ImageNet classes.
        # Removing fc gives us the 2048-dimensional representation.
        self.model = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.model.to(self.device)

        # We are using ResNet only for feature extraction.
        self.model.eval()

        # ImageNet preprocessing associated with the pretrained weights.
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract(self, image_path: str | Path) -> torch.Tensor:
        """
        Extract a feature vector from a single image.

        Returns:
            Tensor with shape [2048].
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        # Add batch dimension:
        # [3, H, W] -> [1, 3, H, W]
        image = image.unsqueeze(0)

        image = image.to(self.device)

        features = self.model(image)

        # ResNet output:
        # [1, 2048, 1, 1]
        features = features.squeeze()

        # Expected:
        # [2048]
        return features.cpu()
