"""Helper functions for plotting and reporting."""
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def plot_training_curves(history, output_dir: Path) -> None:
	"""Persist accuracy and loss plots to disk."""

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	acc = history.history.get("accuracy", [])
	val_acc = history.history.get("val_accuracy", [])
	loss = history.history.get("loss", [])
	val_loss = history.history.get("val_loss", [])
	epochs = range(1, len(acc) + 1)

	plt.figure(figsize=(10, 4))
	plt.subplot(1, 2, 1)
	plt.plot(epochs, acc, label="train")
	plt.plot(epochs, val_acc, label="val")
	plt.title("Accuracy")
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(epochs, loss, label="train")
	plt.plot(epochs, val_loss, label="val")
	plt.title("Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "accuracy_plot.png")
	plt.close()

	plt.figure(figsize=(6, 4))
	plt.plot(epochs, loss, label="train")
	plt.plot(epochs, val_loss, label="val")
	plt.title("Training vs Validation Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "loss_plot.png")
	plt.close()


def save_classification_report(
	y_true: Sequence[int],
	y_pred: Sequence[int],
	target_names: Sequence[str],
	output_dir: Path,
) -> None:
	"""Write a text classification report to disk."""

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	report = classification_report(y_true, y_pred, target_names=target_names)
	(output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
