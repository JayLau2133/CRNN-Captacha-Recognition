import torch
from torch import nn
from tqdm import tqdm

from .config import BLANK_IDX
from .ctc import calculate_metrics, ctc_greedy_decode
from .data import labels_to_strings


def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    total_loss = 0.0
    use_amp = device.type == "cuda"
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

    for images, targets, target_lengths in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            log_probabilities = torch.log_softmax(outputs, dim=2)
            input_lengths = torch.full(
                (images.size(0),), outputs.size(0), dtype=torch.long, device=device
            )
            loss = criterion(log_probabilities, targets, input_lengths, target_lengths)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, device, description="Evaluation"):
    model.eval()
    predicted_strings = []
    true_strings = []
    for images, targets, target_lengths in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predicted_strings.extend(ctc_greedy_decode(outputs.cpu()))
        true_strings.extend(labels_to_strings(targets, target_lengths))
    return calculate_metrics(predicted_strings, true_strings)
