from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
from scipy.special import expit
from datasets import Dataset
import json

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

# --- Metrics ---
def compute_metrics(pred):
    logits, labels = pred
    probabilities = expit(logits)
    predictions = (probabilities > 0.5).astype(int)

    # Calculate F1-macro
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=1)
    return {"f1_macro": f1_macro}

# --- Training Function ---
def train_and_save_model(texts, labels, model_name, output_dir, seed):
    dataset = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    dataset = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=labels.shape[1]
    )

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/subnarrative_model_seed_{seed}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs_subnarrative_seed_{seed}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=3e-5,
        fp16=True,
        logging_steps=50,
        seed=seed,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    model.save_pretrained(f"{output_dir}/subnarrative_model_seed_{seed}")
    tokenizer.save_pretrained(f"{output_dir}/subnarrative_model_seed_{seed}")
    print(f"Model and tokenizer saved to {output_dir}/subnarrative_model_seed_{seed}")

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

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Train multiple models with different seeds
    seeds = [42, 123, 456]
    output_dir = "/content/drive/MyDrive/hindi_subnarrative_models"

    for seed in seeds:
        train_and_save_model(texts, labels, "xlm-roberta-base", output_dir, seed)

    print("All models trained and saved.")
