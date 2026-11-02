import argparse
from pathlib import Path

import torch

from crnn.ctc import ctc_greedy_decode
from crnn.data import preprocess
from crnn.model import CRNN


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize one CAPTCHA image")
    parser.add_argument("--image", type=Path, default=Path("dataset/predict/gig4i.png"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/crnn_best_model.pth")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = CRNN().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    image = preprocess(args.image).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(image)
    print(ctc_greedy_decode(outputs.cpu())[0])


if __name__ == "__main__":
    main()
