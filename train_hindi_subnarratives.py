from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
from scipy.special import expit
from datasets import Dataset
import json
import torch
from torch.nn import BCEWithLogitsLoss

# --- Prepare Subnarrative Labels ---
def prepare_labels_for_subnarratives(training_data, all_labels):
    subnarratives_only = [label for label in all_labels if label["type"] == "S"]
    label_to_idx = {label["label"]: idx for idx, label in enumerate(subnarratives_only)}

    num_classes = len(label_to_idx)
    binary_labels = np.zeros((len(training_data), num_classes))

    for i, article in enumerate(training_data):
        subnarratives = article["subnarratives"]
        indices = [label_to_idx[label] for label in subnarratives if label in label_to_idx]
        binary_labels[i, indices] = 1

    texts = [article["content"] for article in training_data]
    return texts, binary_labels, label_to_idx

# --- Tokenization ---
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

# --- Custom Trainer ---
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


# --- Metrics ---
def compute_metrics(pred):
    logits, labels = pred
    probabilities = expit(logits)
    predictions = (probabilities > 0.5).astype(int)
    f1 = f1_score(labels, predictions, average="samples", zero_division=1)
    return {"f1_samples": f1}

# --- Training with Repeated KFold ---
def train_with_repeated_kfold_and_save(texts, labels):
    dataset = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    dataset = dataset.map(tokenize, batched=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    labels_flat = labels.argmax(axis=1)

    all_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(labels)), labels_flat)):
        print(f"\n=== Fold {fold+1} ===")
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-large", num_labels=labels.shape[1]
        )

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"./logs_fold_{fold}",
            per_device_train_batch_size=8,  # Reduce batch size for memory
            per_device_eval_batch_size=8,
            num_train_epochs=20,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1_samples",
            save_total_limit=1,
            learning_rate=5e-5,
            lr_scheduler_type="linear",
            fp16=True
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train()

        # Compute F1 on validation set
        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions
        probabilities = expit(logits)
        predicted_labels = (probabilities > 0.5).astype(int)

        f1 = f1_score(val_dataset["label"], predicted_labels, average="samples", zero_division=1)
        all_f1_scores.append(f1)
        print(f"F1 Score for fold {fold+1}: {f1}")

    mean_f1 = np.mean(all_f1_scores)
    print(f"\n=== Mean F1 Score (RepeatedStratifiedKFold): {mean_f1} ===")

    # Save the final best model
    final_output_dir = "/content/drive/MyDrive/hindi_xlmrlarge_subnarrative_model"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Subnarrative model and tokenizer saved to {final_output_dir}.")

    # Save final metrics
    with open("/content/drive/MyDrive/hindi_xlmrlarge_subnarrative_model/final_metrics.json", "w") as f:
        json.dump({"mean_f1": mean_f1, "fold_f1_scores": all_f1_scores}, f, indent=4)
    return mean_f1

# --- Main Script ---
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data", "final_hindi_training_dataset.json")
    labels_path = os.path.join(current_dir, "data", "hindi_all_labels.json")

    print("Loading data...")
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)["labels"]

    print("Preparing subnarrative labels...")
    texts, labels, label_to_idx = prepare_labels_for_subnarratives(training_data, all_labels)

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

    print("Training with Repeated Stratified K-Fold and saving the model...")
    train_with_repeated_kfold_and_save(texts, labels)
