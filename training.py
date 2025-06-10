from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from data_utils import prepare_dataset
from dataset import SolarCellDataset
from model import RegressorModel
from model_utils import evaluate, plot_predictions, train_one_epoch, validate


def train_pipeline(
    folder_path,
    img_size=(80, 80),
    val_cells=[
        ("maurizio_test_2", 7),
        ("maurizio_test_2", 16),
        ("maurizio_test_4", 14),
        ("maurizio_test_4", 16),
    ],
    test_cells=[
        ("maurizio_test_2", 5),
        ("maurizio_test_2", 6),
        ("maurizio_test_4", 7),
        ("maurizio_test_4", 11),
    ],
    future_offset=20,
    batch_size=32,
    num_epochs=10,
    lr=1e-2,
    model_save_path="model",
    model_save_name="best_model.pth",
    figure_path="figures",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    model_save_path = Path(model_save_path)
    Path.mkdir(model_save_path, exist_ok=True)

    figure_path = Path(figure_path)
    Path.mkdir(figure_path, exist_ok=True)

    # Step 1: Prepare datasets
    train_pairs, val_pairs, test_pairs = prepare_dataset(
        folder_path,
        val_cells=val_cells,
        test_cells=test_cells,
        future_offset=future_offset,
    )

    train_dataset = SolarCellDataset(
        [(ref, cur, ss) for _, _, ref, cur, ss in train_pairs], img_size=img_size
    )
    val_dataset = SolarCellDataset(
        [(ref, cur, ss) for _, _, ref, cur, ss in val_pairs], img_size=img_size
    )
    test_dataset = SolarCellDataset(
        [(ref, cur, ss) for _, _, ref, cur, ss in test_pairs], img_size=img_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 2: Initialize model, optimizer, loss
    model = RegressorModel().to(device)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, threshold=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.95)  # 5% decay per epoch

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    # Step 3: Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6e}"
        )

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path / model_save_name)
            print(
                f"New best model saved at epoch {epoch+1} with val loss {val_loss:.6f}"
            )

    # Step 4: Plot Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.grid()
    plt.savefig(figure_path / "loss.png", dpi=330, bbox_inches="tight")
    plt.show()

    print("Training complete!")

    # Step 5: Evaluation on Validation and Test Sets
    model.load_state_dict(torch.load(model_save_path / model_save_name))

    train_preds, train_targets = evaluate(model, train_loader, device)
    val_preds, val_targets = evaluate(model, val_loader, device)
    test_preds, test_targets = evaluate(model, test_loader, device)

    # Scatter plots
    plot_predictions(train_targets, train_preds, figure_path, title="train")
    plot_predictions(val_targets, val_preds, figure_path, title="validation")
    plot_predictions(test_targets, test_preds, figure_path, title="test")

    # Final test loss
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")

    return model, train_losses, val_losses
