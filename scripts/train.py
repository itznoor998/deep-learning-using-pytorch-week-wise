"""Minimal training script used for quick experiments."""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.data import get_mnist_loaders, get_cifar10_loaders


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


def train_mnist(args):
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {total/len(train_loader):.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-path', type=str, default='models/checkpoint.pth')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        train_mnist(args)
    else:
        print('CIFAR training not implemented in this minimal script')


if __name__ == '__main__':
    main()
