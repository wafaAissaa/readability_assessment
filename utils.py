import os
import csv

import os, csv

def save_experiment_results(run_name, wandb_config, trainer_state,
                            metrics_val, metrics_test, output_path="results/scores.csv"):
    """
    Save training, validation, and test results to a CSV file.
    """

    # Ensure results directory exists (only if there is a dir component)
    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Define CSV columns
    fieldnames = [
        "run_name", "model_name", "learning_rate", "epoch",
        "batch_size", "weight_decay", "dropout", "OLL_alpha",
        "val_loss", "val_accuracy", "val_f1_macro",
        "test_loss", "test_accuracy", "test_f1_macro"
    ]

    # Build data row (use dict-style consistently)
    row = {
        "run_name": run_name,
        "model_name": wandb_config["model_name"],
        "learning_rate": wandb_config["learning_rate"],
        "epoch": round(trainer_state.epoch, 2) if getattr(trainer_state, "epoch", None) is not None else None,
        "batch_size": wandb_config["batch_size"],
        "weight_decay": wandb_config["weight_decay"],
        "dropout": wandb_config["dropout"],
        "OLL_alpha": wandb_config["OLL_alpha"],   # <-- fixed key
        "val_loss": metrics_val.get("eval_loss"),
        "val_accuracy": metrics_val.get("eval_accuracy"),
        "val_f1_macro": metrics_val.get("eval_f1_macro"),
        "test_loss": metrics_test.get("eval_loss"),
        "test_accuracy": metrics_test.get("eval_accuracy"),
        "test_f1_macro": metrics_test.get("eval_f1_macro"),
    }

    # Append row and add header if file is new
    write_header = not os.path.exists(output_path)
    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # Safe formatting helpers
    def fmt(x): return "NA" if x is None else f"{x:.4f}"

    # Human-readable printout
    print("\n=== Evaluation Summary ===")
    print(f"Model: {row['model_name']}  |  Run: {row['run_name']}")
    print(
        f"Epoch: {row['epoch']}  |  LR: {row['learning_rate']}  |  "
        f"Batch: {row['batch_size']}  |  WD: {row['weight_decay']}  |  "
        f"Dropout: {row['dropout']}  |  OLL_alpha: {row['OLL_alpha']}"
    )
    print(f"Validation -> Loss: {fmt(row['val_loss'])}, Acc: {fmt(row['val_accuracy'])}, F1_macro: {fmt(row['val_f1_macro'])}")
    print(f"Test       -> Loss: {fmt(row['test_loss'])}, Acc: {fmt(row['test_accuracy'])}, F1_macro: {fmt(row['test_f1_macro'])}\n")
