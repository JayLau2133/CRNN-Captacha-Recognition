import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crnn.data import CaptchaDataset, collate_fn
from crnn.engine import evaluate as run_evaluation
from crnn.model import CRNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CRNN CAPTCHA checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/crnn_best_model.pth")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dataset = CaptchaDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = CRNN().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    metrics = run_evaluation(model, loader, device)
    print(f"Samples: {metrics['samples']}")
    print(f"Exact-match accuracy: {metrics['accuracy']:.2%}")
    print(f"Character error rate: {metrics['cer']:.2%}")


if __name__ == "__main__":
    main()
