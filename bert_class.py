import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for command classification")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV file (text,label)")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV file (text,label)")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

# Custom Dataset
torch.manual_seed(42)
class CommandDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="weighted")
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    args = parse_args()

    # Load data
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)

    print(123)

    # Label mapping: 0=mouse/hotkeys, 1=python execution, 2=no action
    # Ensure your CSV uses these integer labels.

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_dataset = CommandDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer
    )
    val_dataset = CommandDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )

    # Data collator to dynamically pad batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()