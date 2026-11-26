"""Train one multilingual model; run from the repository root."""

import argparse
from pathlib import Path

import sacrebleu
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from common import encode, generate, initialize_tokenizer, load_config, read_rows


class EncodedDataset:
    def __init__(self, rows, tokenizer, config):
        self.items = [encode(row, tokenizer, config) for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class MultilingualTrainer(Seq2SeqTrainer):
    def __init__(self, *args, validation_rows, pipeline_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_rows = validation_rows
        self.pipeline_config = pipeline_config

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs
    ):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )
        predictions = generate(
            self.validation_rows,
            self.model,
            self.processing_class,
            self.pipeline_config,
        )
        references = [row["TARGET_TEXT"] for row in self.validation_rows]
        metrics[f"{metric_key_prefix}_chrf"] = sacrebleu.corpus_chrf(
            predictions, [references], word_order=2
        ).score
        metrics[f"{metric_key_prefix}_bleu"] = sacrebleu.corpus_bleu(
            predictions, [references], tokenize="none"
        ).score
        self.log(
            {
                f"{metric_key_prefix}_chrf": metrics[f"{metric_key_prefix}_chrf"],
                f"{metric_key_prefix}_bleu": metrics[f"{metric_key_prefix}_bleu"],
            }
        )
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_nllb.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    train = read_rows(cfg["train_file"], labeled=True)
    validation = read_rows(cfg["validation_file"], labeled=True)
    overlap = {r["VERSE_KEY"] for r in train} & {r["VERSE_KEY"] for r in validation}
    if overlap:
        raise ValueError(
            f"Train/validation share {len(overlap)} verse keys; prevent data leakage"
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    initialize_tokenizer(tokenizer, model, cfg["model_family"])
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["checkpoint_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        seed=cfg["seed"],
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        logging_steps=50,
    )
    trainer = MultilingualTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=EncodedDataset(train, tokenizer, cfg),
        eval_dataset=EncodedDataset(validation, tokenizer, cfg),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        validation_rows=validation,
        pipeline_config=cfg,
    )
    trainer.train()
    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg["model_dir"])
    tokenizer.save_pretrained(cfg["model_dir"])
    print(f"Best model saved to {cfg['model_dir']}")


if __name__ == "__main__":
    main()
