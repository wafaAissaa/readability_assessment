from transformers import CamembertForSequenceClassification
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import pandas as pd
import torch
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer
import wandb
from transformers import Trainer, TrainingArguments, set_seed
import argparse
import os

import torch.nn.functional as F

set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

distributions = None


def get_distributions(data_path):
    global distributions
    input_file = pd.read_csv(data_path, index_col='text_indice')
    labels = input_file[['gold_score_20_label']]
    counts = labels.value_counts(normalize=True) * 100
    distributions = [float(counts.loc[l].values[0]) for l in label_to_id.keys()]


label_to_id = {
    'Très Facile': 0,
    'Facile': 1,
    'Accessible': 2,
    '+Complexe': 3
}

id_to_label = {
    0: 'Très Facile',
    1: 'Facile',
    2: 'Accessible',
    3: '+Complexe'
}


def load_data(data_path):
    # load csv file
    dataset = load_dataset('csv', data_files=data_path)

    label_column_name = 'gold_score_20_label'
    dataset = dataset.map(lambda examples: {'labels': label_to_id[examples[label_column_name]]})

    return dataset  # dataset contains only a train but it will be splitted later


# load CamemBERT
def load_model(model_name, dropout):

    model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=4,
                                                                   hidden_dropout_prob=dropout,
                                                                   attention_probs_dropout_prob=dropout)

    return model


def get_model_init_function(model_name, dropout):
    def model_init():
        model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=4,
                                                                   hidden_dropout_prob=dropout,
                                                                   attention_probs_dropout_prob=dropout)
        torch.nn.init.xavier_uniform_(model.classifier.dense.weight)
        torch.nn.init.xavier_uniform_(model.classifier.out_proj.weight)
        torch.nn.init.zeros_(model.classifier.dense.bias)
        torch.nn.init.zeros_(model.classifier.out_proj.bias)
        return model
    return model_init


def load_fold(dataset, fold_id):
    """
    Returns fold train, validation, and test datasets.
    """
    folds_indexes_dico = load_fold_indexes('../data')
    train_splits_index = folds_indexes_dico['train_splits_index']
    val_splits_index = folds_indexes_dico['val_splits_index']
    test_splits_index = folds_indexes_dico['test_splits_index']

    # Convert to pandas dataframe for stratified split
    df = pd.DataFrame(dataset["train"])
    df.set_index("text_indice", inplace=True)

    train_df = df.loc[train_splits_index[fold_id]]
    val_df = df.loc[val_splits_index[fold_id]]
    test_df = df.loc[test_splits_index[fold_id]]

    # Convert pandas dataframes back to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Return dataset splits as a DatasetDict
    return DatasetDict({'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})


tokenizer = None
def initialize_tokenizer(model_name):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenization of the full dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


# evaluate on classification
metric_classification = evaluate.load("accuracy")

def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_classification.compute(predictions=predictions, references=labels)


# Custom trainer to overwrite the loss function
class CustomClassificationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss_cross_entropy_weighted(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Custom loss function for training.
        By default, Trainer uses CrossEntropyLoss for classification.
        This can be overridden here for custom loss.
        """
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)
        logits = outputs.logits

        # inverse of the distributions
        inverse_distributions = [1 / x for x in distributions]
        # normalize the inverse distributions
        inverse_distributions = [x / sum(inverse_distributions) for x in
                                 inverse_distributions]
        # Custom loss function:
        class_weights = torch.tensor([inverse_distributions]).to(
            logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.squeeze())
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Custom loss function for training in a ordinal_weighted classification settings: Ordinal Log-Loss (OLL).
        By default, Trainer uses CrossEntropyLoss for classification.
        This can be overridden here for custom loss.
        """
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)
        logits = outputs.logits

        alpha = 1.5  # [1, 1.5, 2]

        logits = model(**inputs).logits
        probas = F.softmax(logits, dim=1)
        dist_matrix = [[0, 1, 2, 3], [1, 0, 1, 3], [2, 1, 0, 1], [3, 2, 1, 0]]  # distance entre les classes
        true_labels = [4 * [labels[k].item()] for k in range(len(labels))]  # 4 nb classe
        label_ids = len(labels) * [[k for k in range(4)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(4)] for j in
                     range(len(labels))]
        distances_tensor = torch.tensor(distances, device='cuda:0', requires_grad=True)

        err = -torch.log(1 - probas) * abs(distances_tensor) ** alpha

        # inverse of the distributions
        inverse_distributions = [1 / x for x in distributions]
        # normalize the inverse distributions
        inverse_distributions = [x / sum(inverse_distributions) for x in
                                 inverse_distributions]

        class_weights = torch.tensor([x / sum(inverse_distributions) for x in inverse_distributions],
                                     device=logits.device)
        sample_weights = class_weights[labels]

        weighted_err = torch.sum(err, dim=1) * sample_weights

        # loss = torch.sum(err,axis=1).mean()
        loss = weighted_err.mean()

        # loss.backward()

        return (loss, outputs) if return_outputs else loss


class CustomRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Custom loss function for training.
        By default, Trainer uses MSE for regression.
        This can be overridden here for custom loss.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(logits.squeeze(), labels)

        return (loss, outputs) if return_outputs else loss


def train_and_evaluate(task, data_path, fold_id=0, ):
    """
        Train and evaluate a model with a given set of hyperparameters.
    """
    wandb.login()  # not useful when we already have the API key
    wandb.init(project="DL_ordinal_weighted_test", entity="iRead4skills")
    run_name = "name:%s_lr:%s_bs:%s_fold:%s_wd:%s_drop:%s" % (wandb.config.model_name, wandb.config.learning_rate,
                                                              wandb.config.batch_size, fold_id,
                                                              wandb.config.weight_decay, wandb.config.dropout)

    folder_path = '/globalscratch/ucl/cental/troux/models_ordinal_weighted/' + run_name

    if os.path.isdir(folder_path):
        print(f"The folder '{folder_path}' exists. Training skipped")
        wandb.run.name = run_name + "delete"
        wandb.finish()
        return

    wandb.run.name = run_name
    # Load CamemBERT model
    model_init_fn = get_model_init_function(task, wandb.config.model_name, wandb.config.dropout)
    # Load data
    dataset = load_data(data_path, task)
    # Load CamemBERT tokenizer
    get_distributions(data_path)
    initialize_tokenizer(wandb.config.model_name)
    # Apply tokenization to dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    fold_dataset = load_fold(tokenized_datasets, fold_id)
    train_fold = fold_dataset['train']
    eval_fold = fold_dataset['eval']
    test_fold = fold_dataset['test']

    # List all columns except the ones you want to keep
    columns_to_remove = [col for col in train_fold.column_names if col not in ['input_ids', 'attention_mask', 'labels']]

    # Remove the unnecessary columns from each split
    train_dataset = train_fold.remove_columns(columns_to_remove)
    eval_dataset = eval_fold.remove_columns(columns_to_remove)
    test_dataset = test_fold.remove_columns(columns_to_remove)

    if task == 'classification':
        output_dir = '/globalscratch/ucl/cental/troux/models_ordinal_weighted/' + run_name

    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # instead of accuracy
        seed=42,
        report_to="wandb",
        learning_rate=wandb.config["learning_rate"],
        num_train_epochs=wandb.config["epochs"],
        per_device_train_batch_size=wandb.config["batch_size"],
        per_device_eval_batch_size=wandb.config["batch_size"],
        weight_decay=wandb.config["weight_decay"],
    )
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=wandb.config["patience"])

    # print("Train dataset:", train_dataset)
    trainer = CustomClassificationTrainer(
        model=None,
        model_init=model_init_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_classification,
        callbacks=[early_stopping_callback]
    )

    # Train model
    trainer.train()

    # Evaluation on validation
    metrics_val = trainer.evaluate()
    print("Validation Results : ", metrics_val)

    eval_type = 'eval_accuracy'

    wandb.log({'val_accuracy': metrics_val[eval_type], 'val_loss': metrics_val['eval_loss']})
    # Evaluation on test
    metrics_test = trainer.evaluate(test_dataset)
    print("Test Results : ", metrics_test)
    wandb.log({'Test_accuracy': metrics_test[eval_type], 'Test_loss': metrics_test['eval_loss']})

    # Save scores accorded to the model
    with open("results/all_DL_scores_ordinal_weighted_" + str(fold_id) + ".txt", "a") as f:
        params_list = str(wandb.config["learning_rate"]) + "," + str(trainer.state.epoch) + "," + str(
            wandb.config["batch_size"]) + "," + str(wandb.config["weight_decay"]) + "," + str(
            wandb.config["dropout"])
        f.write(run_name + "," + wandb.config.model_name + "," + str(fold_id) + "," + params_list + "," + str(
            metrics_val['eval_loss']) + "," + str(metrics_val[eval_type]) + "," + str(
            metrics_test['eval_loss']) + "," + str(metrics_test[eval_type]) + "\n")

    wandb.finish()

    return metrics_val, metrics_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_id', type=int, default=0, help="add the fold id 0 to 4")
    args = parser.parse_args()

    data_path = './data/Qualtrics_Annotations_B.csv'

    # Hyperparameter Grid
    # sweep_configuration = {
    #     "name": "launch",
    #     "method": "grid",
    #     "run_cap": 80,
    #     "metric": {"goal": "maximize", "name": "val_accuracy"},
    #     "parameters": {
    #         "model_name": {"values": ['camembert-base', 'almanach/camembertv2-base', 'dangvantuan/sentence-camembert-base']},
    #         "learning_rate": {"values": [1e-5, 1e-4]}, # , 1e-3
    #         "batch_size": {"values": [16, 32, 64]},
    #         "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
    #         "dropout": {"values": [0.1]}, # , 0.3, 0.5
    #         "epochs": {"value": 200},
    #         "patience": {"value": 10}
    #     },
    # }

    sweep_configuration = {
        "name": "launch",
        "method": "grid",
        "run_cap": 80,
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "model_name": {
                "values": ['camembert-base', 'almanach/camembertv2-base', 'dangvantuan/sentence-camembert-base']},
            "learning_rate": {"values": [1e-5, 1e-4]},  # , 1e-3
            "batch_size": {"values": [16, 32, 64]},
            "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "dropout": {"values": [0.1]},  # , 0.3, 0.5
            "epochs": {"value": 200},
            "patience": {"value": 10}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="DL_ordinal_weighted_test", entity="iRead4skills")

    wandb.agent(sweep_id, function=lambda: train_and_evaluate(data_path, fold_id=args.fold_id))