import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from crnn.data import CaptchaDataset, collate_fn
from crnn.engine import evaluate, train_one_epoch
from crnn.model import CRNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CRNN CAPTCHA model")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, batch_size, workers, shuffle, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
        persistent_workers=workers > 0,
    )


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_accuracy):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_acc": best_accuracy,
    }, path)


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_interval <= 0:
        raise ValueError("Epochs, batch size and evaluation interval must be positive")
    if args.workers < 0:
        raise ValueError("Workers must be non-negative")

    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "crnn_checkpoint.pth"
    best_model_path = args.output_dir / "crnn_best_model.pth"

    train_dataset = CaptchaDataset(args.data_dir / "train")
    test_dataset = CaptchaDataset(args.data_dir / "test")
    train_loader = build_loader(
        train_dataset, args.batch_size, args.workers, True, device, args.seed
    )
    test_loader = build_loader(
        test_dataset, args.batch_size * 2, args.workers, False, device, args.seed
    )

    model = CRNN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_accuracy = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_acc"]
        print(
            f"Resumed {args.resume}: epoch {checkpoint['epoch']}, "
            f"best accuracy {best_accuracy:.2%}"
        )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}; test samples: {len(test_dataset)}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step(train_loss)
        if epoch % args.eval_interval != 0 and epoch != args.epochs:
            continue

        metrics = evaluate(model, test_loader, device, description=f"Epoch {epoch:03d} [test]")
        print(
            f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
            f"Accuracy: {metrics['accuracy']:.2%} | CER: {metrics['cer']:.2%}"
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)
        save_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, epoch, best_accuracy
        )

    if start_epoch > args.epochs:
        metrics = evaluate(model, test_loader, device)
        print(f"Accuracy: {metrics['accuracy']:.2%} | CER: {metrics['cer']:.2%}")
    print(f"Best exact-match accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
