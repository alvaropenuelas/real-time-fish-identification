"""
Training loop: cosine LR decay, AdamW, early stopping, checkpointing.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def _inference_mode(model):
    # Sets model to inference mode (disables dropout, batchnorm uses running stats)
    # nn.Module.eval() — not the Python builtin
    model.eval()


def train(model, train_loader, val_loader, num_epochs, output_dir, device, lr=1e-3, patience=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.to(device)
    best_val_acc, epochs_no_improve, history = 0.0, 0, []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        train_acc = correct / total

        _inference_mode(model)
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_loss += criterion(out, labels).item() * images.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:03d}/{num_epochs}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping (best val_acc={best_val_acc:.4f})")
                break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best val_acc: {best_val_acc:.4f} → {output_dir}/best_model.pt")
    return best_val_acc
