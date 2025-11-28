# -*- coding: utf-8 -*-
"""
Example training script for the RoBERTaforNER model.

This script loads a dataset (e.g., conll2003), tokenizes it,
and then uses the Hugging Face Trainer to fine-tune the
custom model defined in `model.py`.

Usage:
    python train.py --model_name "xlm-roberta-large" \
                    --dataset_name "conll2003" \
                    --output_dir "./ner_model" \
                    --use_bilstm
"""

import argparse
import datasets
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    XLMRobertaConfig
)

# Import custom model class
from model import RoBERTaforNER

# Metrics
seqeval = evaluate.load("seqeval")
label_list = datasets.load_dataset("conll2003", split="train").features["ner_tags"].feature.names

def compute_metrics(p):
    """
    Computes F1, precision, recall, and accuracy for NER.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main(args):
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    # Load Dataset
    raw_datasets = load_dataset(args.dataset_name)

    # Preprocess Data
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load Model
    num_labels = raw_datasets["train"].features["ner_tags"].feature.num_classes

    # Load the configuration
    config = XLMRobertaConfig.from_pretrained(args.model_name, num_labels=num_labels)

    print(f"Loading custom model RoBERTaforNER...")
    print(f"Using BiLSTM: {args.use_bilstm}")

    # Instantiate our custom model
    model = RoBERTaforNER(
        config=config,
        num_labels=num_labels,
        use_bilstm=args.use_bilstm,
        lstm_hidden_size=args.lstm_hidden_size,
        num_lstm_layers=args.lstm_layers
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate
    print("Starting final evaluation on test set...")
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    print(eval_results)

    # Save Model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom RoBERTa-NER model.")

    parser.add_argument("--model_name", type=str, default="xlm-roberta-large", help="Base pre-trained model name.")
    parser.add_argument("--dataset_name", type=str, default="conll2003", help="Dataset name from Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./ner-model", help="Directory to save the trained model.")

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)

    parser.add_argument("--use_bilstm", action="store_true", help="Include this flag to use the BiLSTM layer.")
    parser.add_argument("--lstm_hidden_size", type=int, default=1024, help="Hidden size for the BiLSTM.")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of layers for the BiLSTM.")

    args = parser.parse_args()
    main(args)
