from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split


class Flickr8kPreprocessor:
    """
    Handles loading, validating, and splitting the Flickr8k dataset.
    """

    def __init__(
        self,
        dataset_dir: str = "data/raw/Flickr8k",
        random_seed: int = 42,
        test_size: float = 0.10,
        val_size: float = 0.10,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "Images"
        self.captions_file = self.dataset_dir / "captions.txt"

        self.random_seed = random_seed
        self.test_size = test_size
        self.val_size = val_size

    def load_captions(self) -> pd.DataFrame:
        """Load the caption CSV file."""

        if not self.captions_file.exists():
            raise FileNotFoundError(
                f"Captions file not found: {self.captions_file}"
            )

        df = pd.read_csv(self.captions_file)

        required_columns = {"image", "caption"}

        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Expected columns {required_columns}, "
                f"but found {set(df.columns)}"
            )

        return df

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate the basic integrity of the dataset."""

        # Check missing values
        if df["image"].isnull().any():
            raise ValueError("Dataset contains missing image names.")

        if df["caption"].isnull().any():
            raise ValueError("Dataset contains missing captions.")

        # Check that every image has exactly five captions
        captions_per_image = df.groupby("image").size()

        invalid_images = captions_per_image[
            captions_per_image != 5
        ]

        if len(invalid_images) > 0:
            raise ValueError(
                f"{len(invalid_images)} images do not have exactly "
                f"5 captions."
            )

        # Check that every referenced image exists
        actual_images = {
            path.name
            for path in self.images_dir.iterdir()
            if path.is_file()
        }

        caption_images = set(df["image"])

        missing_images = caption_images - actual_images

        if missing_images:
            raise ValueError(
                f"{len(missing_images)} images referenced in captions.txt "
                f"are missing from the Images directory."
            )

    def create_image_caption_mapping(
        self,
        df: pd.DataFrame,
    ) -> dict[str, list[str]]:
        """
        Create:

            image_name -> list of five captions
        """

        image_captions = defaultdict(list)

        for _, row in df.iterrows():
            image_captions[row["image"]].append(row["caption"])

        return dict(image_captions)

    def split_images(
        self,
        image_names: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Split images into train, validation, and test sets.

        The split happens at IMAGE level so that all captions
        belonging to an image stay in the same split.
        """

        train_val, test = train_test_split(
            image_names,
            test_size=self.test_size,
            random_state=self.random_seed,
        )

        # val_size is relative to the remaining train+validation set.
        relative_val_size = self.val_size / (1.0 - self.test_size)

        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=self.random_seed,
        )

        return train, val, test

    @staticmethod
    def check_no_data_leakage(
        train: list[str],
        val: list[str],
        test: list[str],
    ) -> None:
        """Verify that no image occurs in multiple splits."""

        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        if train_set & val_set:
            raise ValueError("Data leakage detected between train and val.")

        if train_set & test_set:
            raise ValueError("Data leakage detected between train and test.")

        if val_set & test_set:
            raise ValueError("Data leakage detected between val and test.")

    def prepare(self):
        """
        Complete dataset preparation pipeline.

        Returns:
            image_captions
            train_images
            val_images
            test_images
        """

        df = self.load_captions()

        self.validate_dataset(df)

        image_captions = self.create_image_caption_mapping(df)

        image_names = list(image_captions.keys())

        train_images, val_images, test_images = self.split_images(
            image_names
        )

        self.check_no_data_leakage(
            train_images,
            val_images,
            test_images,
        )

        return (
            image_captions,
            train_images,
            val_images,
            test_images,
        )
        
        
    def get_captions_for_images(
        self,
        image_captions: dict[str, list[str]],
        image_names: list[str],
         ) -> list[str]:
        """
        Return all captions belonging to the given images.
        """

        captions = []

        for image_name in image_names:
            captions.extend(image_captions[image_name])

        return captions
