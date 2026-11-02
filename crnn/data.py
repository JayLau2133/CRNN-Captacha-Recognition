from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from .config import CHAR_SET, IMAGE_SIZE


def preprocess(image_path):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, IMAGE_SIZE)
    image = (image / 255.0 - 0.5) / 0.5
    return torch.from_numpy(image).float().unsqueeze(0)


class CaptchaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        self.files = sorted(
            path for path in self.root_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if not self.files:
            raise ValueError(f"No PNG images found in: {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        image = preprocess(image_path)
        label_string = image_path.stem
        try:
            label = [CHAR_SET.index(character) for character in label_string]
        except ValueError as error:
            raise ValueError(
                f"Label {label_string!r} contains a character outside CHAR_SET"
            ) from error
        if not label:
            raise ValueError(f"Empty label for image: {image_path}")
        return image, torch.tensor(label, dtype=torch.long), len(label)


def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.cat(labels, dim=0),
        torch.tensor(label_lengths, dtype=torch.long),
    )


def labels_to_strings(targets, target_lengths):
    strings = []
    offset = 0
    for length in target_lengths.tolist():
        indices = targets[offset:offset + length].tolist()
        strings.append("".join(CHAR_SET[index] for index in indices))
        offset += length
    return strings
