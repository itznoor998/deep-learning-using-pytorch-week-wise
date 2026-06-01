"""Minimal evaluation script to load a checkpoint and run on test set."""
import argparse
import torch
import torch.nn as nn
from scripts.data import get_mnist_loaders


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(args):
    _, _, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"Test accuracy: {correct/total:.4f} ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
