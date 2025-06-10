import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, target in dataloader:  # <-- inputs is (B, 2, H, W)
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(inputs)  # <-- model expects only 1 input
        loss = criterion(output, target.squeeze(1))  # be careful with shapes!
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs)
            loss = criterion(output, target.squeeze(1))

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss


def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)

            preds.append(output.cpu())
            targets.append(target.cpu())

    preds = torch.cat(preds).squeeze().numpy()
    targets = torch.cat(targets).squeeze().numpy()
    return preds, targets


def plot_predictions(targets, predictions, figure_path, title="validation"):
    r2 = r2_score(targets, predictions)

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, predictions, alpha=0.6, edgecolors="k")
    plt.plot(
        [targets.min(), targets.max()],
        [targets.min(), targets.max()],
        "r--",
        label="Ideal",
    )
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(f"Predicted vs True ({title})\n$R^2$: {r2:.3f}")
    plt.grid()
    plt.savefig(figure_path / f"{title}_parity.png", dpi=330, bbox_inches="tight")
    plt.show()
