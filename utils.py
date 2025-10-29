import os
import csv

def save_experiment_results(run_name, wandb_config, trainer_state,
                            metrics_val, metrics_test, output_dir="results/scores.csv"):
    """
    Save training, validation, and test results to a CSV file.

    Parameters
    ----------
    run_name : str
        Name or ID of the W&B run.
    wandb_config : dict
        W&B configuration dictionary (wandb.config).
    trainer_state : TrainerState
        Hugging Face Trainer state (trainer.state).
    metrics_val : dict
        Metrics from trainer.evaluate() on validation set.
    metrics_test : dict
        Metrics from trainer.evaluate() on test set.
    output_dir : str, optional
        Path to the CSV file where results will be appended.
    """

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Define CSV columns
    fieldnames = [
        "run_name", "model_name", "learning_rate", "epoch",
        "batch_size", "weight_decay", "dropout", "OLL_alpha"
        "val_loss", "val_accuracy", "val_f1_macro",
        "test_loss", "test_accuracy", "test_f1_macro"
    ]

    # Build data row
    row = {
        "run_name": run_name,
        "model_name": wandb_config.model_name,
        "learning_rate": wandb_config["learning_rate"],
        "epoch": round(trainer_state.epoch, 2) if trainer_state.epoch else None,
        "batch_size": wandb_config["batch_size"],
        "weight_decay": wandb_config["weight_decay"],
        "dropout": wandb_config["dropout"],
        "alpha" : wandb_config["OLL_alpha"],
        "val_loss": metrics_val.get("eval_loss"),
        "val_accuracy": metrics_val.get("eval_accuracy"),
        "val_f1_macro": metrics_val.get("eval_f1_macro"),
        "test_loss": metrics_test.get("eval_loss"),
        "test_accuracy": metrics_test.get("eval_accuracy"),
        "test_f1_macro": metrics_test.get("eval_f1_macro"),
    }

    # Append row and add header if file is new
    write_header = not os.path.exists(output_dir)
    with open(output_dir, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # Optional: human-readable printout
    print("\n=== Evaluation Summary ===")
    print(f"Model: {row['model_name']}  |  Run: {row['run_name']}")
    print(f"Epoch: {row['epoch']}  |  LR: {row['learning_rate']}  |  "
          f"Batch: {row['batch_size']}  |  WD: {row['weight_decay']}  |  Dropout: {row['dropout']} |  OLL_alpha: {row['OLL_alpha']}")
    print(f"Validation -> Loss: {row['val_loss']:.4f}, Acc: {row['val_accuracy']:.4f}, F1_macro: {row['val_f1_macro']:.4f}")
    print(f"Test       -> Loss: {row['test_loss']:.4f}, Acc: {row['test_accuracy']:.4f}, F1_macro: {row['test_f1_macro']:.4f}\n")