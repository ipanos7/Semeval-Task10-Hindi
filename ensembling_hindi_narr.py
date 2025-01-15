from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score, classification_report
from scipy.special import expit
from datasets import Dataset
import json,torch


# --- Prepare Narrative Labels ---
def prepare_labels_for_narratives(training_data, all_labels):
    narratives_only = [label for label in all_labels if label["type"] == "N"]
    label_to_idx = {label["label"]: idx for idx, label in enumerate(narratives_only)}

    num_classes = len(label_to_idx)
    binary_labels = np.zeros((len(training_data), num_classes))

    for i, article in enumerate(training_data):
        narratives = article["narratives"]
        indices = [label_to_idx[label] for label in narratives if label in label_to_idx]
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

    # Calculate F1-macro for coarse labels
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=1)

    # F1 samples
    f1_samples = f1_score(labels, predictions, average="samples", zero_division=1)

    # Detailed Classification Report (Optional)
    report = classification_report(labels, predictions, output_dict=True, zero_division=1)

    return {
        "f1_macro": f1_macro,
        "f1_samples": f1_samples,
        "classification_report": report,
    }

# --- Training Function ---
def train_and_save_model(texts, labels, model_name, output_dir, seed):
    dataset = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    dataset = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=labels.shape[1]
    )

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/narrative_model_seed_{seed}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs_narrative_seed_{seed}",
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

    model.save_pretrained(f"{output_dir}/narrative_model_seed_{seed}")
    tokenizer.save_pretrained(f"{output_dir}/narrative_model_seed_{seed}")
    print(f"Model and tokenizer saved to {output_dir}/narrative_model_seed_{seed}")

# --- Evaluate Ensemble ---
def evaluate_ensemble(models, tokenizer, dev_texts, dev_labels, label_to_idx):
    probabilities = []

    for model_path in models:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.to("cuda")

        inputs = tokenizer(dev_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities.append(expit(logits.cpu().numpy()))

    # Average probabilities across models
    averaged_probabilities = np.mean(probabilities, axis=0)
    predictions = (averaged_probabilities > 0.5).astype(int)

    # Calculate metrics
    f1_macro = f1_score(dev_labels, predictions, average="macro", zero_division=1)
    f1_samples = f1_score(dev_labels, predictions, average="samples", zero_division=1)

    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Samples: {f1_samples}")

    return {
        "f1_macro": f1_macro,
        "f1_samples": f1_samples,
        "predictions": predictions,
    }

# --- Main Script ---
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data", "final_hindi_training_dataset.json")
    labels_path = os.path.join(current_dir, "data", "hindi_all_labels.json")
    dev_data_path = os.path.join(current_dir, "data", "hindi_dev_dataset.json")

    print("Loading data...")
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)["labels"]
    with open(dev_data_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    print("Preparing narrative labels...")
    texts, labels, label_to_idx = prepare_labels_for_narratives(training_data, all_labels)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Train multiple models with different seeds
    seeds = [42, 123, 456]
    output_dir = "/content/drive/MyDrive/hindi_narrative_models"

    for seed in seeds:
        train_and_save_model(texts, labels, "xlm-roberta-base", output_dir, seed)

    print("All models trained and saved.")