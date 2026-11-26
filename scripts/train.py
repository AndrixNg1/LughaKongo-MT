"""Train one multilingual model; run from the repository root."""

import argparse
from pathlib import Path

import sacrebleu
import torch
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
    def __init__(self, *args, validation_rows, pipeline_config, drive_backup_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_rows = validation_rows
        self.pipeline_config = pipeline_config
        self.drive_backup_dir = drive_backup_dir
        self.best_backup_score = float("-inf")

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
        direction_metrics = {}
        for src, tgt in sorted({(r["SOURCE_LANG"], r["TARGET_LANG"]) for r in self.validation_rows}):
            indices = [i for i, r in enumerate(self.validation_rows)
                       if (r["SOURCE_LANG"], r["TARGET_LANG"]) == (src, tgt)]
            direction_metrics[f"{metric_key_prefix}_chrf_{src}_to_{tgt}"] = sacrebleu.corpus_chrf(
                [predictions[i] for i in indices], [[references[i] for i in indices]], word_order=2
            ).score
        print("chrF++ global:", round(metrics[f"{metric_key_prefix}_chrf"], 2))
        for key, score in direction_metrics.items():
            print(f"{key}: {score:.2f}")
        metrics.update(direction_metrics)
        self.log(
            {
                f"{metric_key_prefix}_chrf": metrics[f"{metric_key_prefix}_chrf"],
                f"{metric_key_prefix}_bleu": metrics[f"{metric_key_prefix}_bleu"],
            }
        )
        score = metrics[f"{metric_key_prefix}_chrf"]
        if self.drive_backup_dir and self.is_world_process_zero() and score > self.best_backup_score:
            # Separate directories preserve earlier backups if a Drive write fails.
            destination = Path(self.drive_backup_dir) / f"best_model_step_{self.state.global_step}"
            if destination.exists():
                raise FileExistsError(f"Backup already exists: {destination}; choose a new run directory")
            self.save_model(str(destination))
            self.processing_class.save_pretrained(destination)
            self.best_backup_score = score
            print(f"Drive backup: {destination} (chrF++ {score:.2f})")
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_nllb.json")
    parser.add_argument("--drive_backup_dir", default=None)
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
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    use_fp16 = "T4" in gpu_name.upper() and cfg["model_family"] == "nllb"
    print(f"Device: {gpu_name}; fp16: {use_fp16}")
    checkpointing = cfg.get("gradient_checkpointing", False)
    if checkpointing:
        model.config.use_cache = False
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
        fp16=use_fp16,
        gradient_checkpointing=checkpointing,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        label_smoothing_factor=cfg.get("label_smoothing_factor", 0.0),
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
        drive_backup_dir=args.drive_backup_dir or cfg.get("drive_backup_dir"),
    )
    trainer.train()
    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg["model_dir"])
    tokenizer.save_pretrained(cfg["model_dir"])
    print(f"Best model saved to {cfg['model_dir']}")


if __name__ == "__main__":
    main()
