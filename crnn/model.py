from torch import nn

from .config import CNN_OUTPUT_HEIGHT, NUM_CLASSES


class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
        )
        self.rnn = nn.LSTM(
            input_size=256 * CNN_OUTPUT_HEIGHT,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(512, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, height, width = x.size()
        if height != CNN_OUTPUT_HEIGHT:
            raise ValueError(
                f"Expected CNN output height {CNN_OUTPUT_HEIGHT}, got {height}. "
                "Use 60x160 input images to stay compatible with the checkpoint."
            )
        x = x.view(batch_size, channels * height, width).permute(2, 0, 1)
        x, _ = self.rnn(x)
        return self.fc(x)
