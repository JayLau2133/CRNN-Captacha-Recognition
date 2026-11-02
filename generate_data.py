import argparse
import random
from pathlib import Path

from captcha.image import ImageCaptcha

from crnn.config import CHAR_SET, IMAGE_HEIGHT, IMAGE_WIDTH


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic CAPTCHA images")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=200)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def existing_labels(split_dir):
    if not split_dir.exists():
        return set()
    return {path.stem for path in split_dir.glob("*.png")}


def generate_split(split_dir, target_count, used_labels, generator, rng, length):
    split_dir.mkdir(parents=True, exist_ok=True)
    current_count = len(list(split_dir.glob("*.png")))
    while current_count < target_count:
        text = "".join(rng.choices(CHAR_SET, k=length))
        if text in used_labels:
            continue
        generator.write(text, str(split_dir / f"{text}.png"))
        used_labels.add(text)
        current_count += 1
    return current_count


def main():
    args = parse_args()
    if args.num_train < 0 or args.num_test < 0:
        raise ValueError("Image counts must be non-negative")
    if args.length <= 0:
        raise ValueError("CAPTCHA length must be positive")
    if args.num_train + args.num_test > len(CHAR_SET) ** args.length:
        raise ValueError("Requested more unique labels than the character space allows")

    rng = random.Random(args.seed)
    generator = ImageCaptcha(width=args.width, height=args.height)
    train_dir = args.output_dir / "train"
    test_dir = args.output_dir / "test"
    train_labels = existing_labels(train_dir)
    test_labels = existing_labels(test_dir)
    overlap = train_labels & test_labels
    if overlap:
        raise ValueError(
            f"Train/test label overlap detected ({len(overlap)} labels); "
            "use separate clean directories to avoid data leakage"
        )
    used_labels = train_labels | test_labels

    train_count = generate_split(
        train_dir, args.num_train, used_labels, generator, rng, args.length
    )
    test_count = generate_split(
        test_dir, args.num_test, used_labels, generator, rng, args.length
    )
    print(f"Train images: {train_count} ({train_dir})")
    print(f"Test images: {test_count} ({test_dir})")


if __name__ == "__main__":
    main()
