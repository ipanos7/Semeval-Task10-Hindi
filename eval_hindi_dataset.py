from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from scipy.special import expit
import numpy as np
import json,torch

def ensemble_predictions(models, tokenizer, texts):
    probabilities = []
    for model_path in models:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.to("cuda")

        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities.append(expit(logits.cpu().numpy()))

    return np.mean(probabilities, axis=0)

def postprocess_predictions(probabilities, threshold=0.5):
    return (probabilities > threshold).astype(int)

def evaluate_ensemble(models_narrative, models_subnarrative, tokenizer, dev_texts, dev_labels, output_file):
    narrative_probs = ensemble_predictions(models_narrative, tokenizer, dev_texts)
    subnarrative_probs = ensemble_predictions(models_subnarrative, tokenizer, dev_texts)

    narrative_preds = postprocess_predictions(narrative_probs)
    subnarrative_preds = postprocess_predictions(subnarrative_probs)

    f1_macro_narrative = f1_score(dev_labels["narratives"], narrative_preds, average="macro", zero_division=1)
    f1_macro_subnarrative = f1_score(dev_labels["subnarratives"], subnarrative_preds, average="macro", zero_division=1)

    print(f"F1 Macro (Narrative): {f1_macro_narrative}")
    print(f"F1 Macro (Subnarrative): {f1_macro_subnarrative}")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(dev_texts):
            f.write(f"{text}\t{narrative_preds[i]}\t{subnarrative_preds[i]}\n")

if __name__ == "__main__":
    dev_data_path = "data/hindi_dev_dataset.json"
    narrative_model_paths = ["/content/drive/MyDrive/hindi_narrative_models/narrative_model_seed_42",
                             "/content/drive/MyDrive/hindi_narrative_models/narrative_model_seed_123",
                             "/content/drive/MyDrive/hindi_narrative_models/narrative_model_seed_456"]
    subnarrative_model_paths = ["/content/drive/MyDrive/hindi_subnarrative_models/subnarrative_model_seed_42",
                                "/content/drive/MyDrive/hindi_subnarrative_models/subnarrative_model_seed_123",
                                "/content/drive/MyDrive/hindi_subnarrative_models/subnarrative_model_seed_456"]
    output_file = "/content/drive/MyDrive/hindi_submission.txt"

    with open(dev_data_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    dev_texts = [article["content"] for article in dev_data]
    dev_labels = {"narratives": [article["narratives"] for article in dev_data],
                  "subnarratives": [article["subnarratives"] for article in dev_data]}

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    evaluate_ensemble(narrative_model_paths, subnarrative_model_paths, tokenizer, dev_texts, dev_labels, output_file)
