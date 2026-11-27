"""Train one multilingual model; run from the repository root."""

import argparse
import json
import math
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
from transformers.trainer_utils import get_last_checkpoint


class EncodedDataset:
    def __init__(self, rows, tokenizer, config):
        self.items = [encode(row, tokenizer, config) for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TranslationCollator(DataCollatorForSeq2Seq):
    """Supply decoder inputs when M2M100 lacks the label-shifting helper."""

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        if (self.model is not None and self.model.config.model_type == "m2m_100"
                and "labels" in batch and "decoder_input_ids" not in batch):
            labels = batch["labels"]
            pad_id = self.model.config.pad_token_id
            start_id = self.model.config.decoder_start_token_id
            if pad_id is None or start_id is None:
                raise ValueError("The model must define pad_token_id and decoder_start_token_id")
            decoder_inputs = labels.new_full(labels.shape, pad_id)
            decoder_inputs[:, 0] = start_id
            decoder_inputs[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad_id)
            batch["decoder_input_ids"] = decoder_inputs
        return batch


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
        try:
            predictions = list(generate(
                self.validation_rows,
                self.model,
                self.processing_class,
                self.pipeline_config,
            ))
            if len(predictions) != len(self.validation_rows) or any(not isinstance(p, str) for p in predictions):
                raise ValueError("Predictions must contain one string per validation row")
        except Exception as exc:
            raise RuntimeError(f"Validation generation failed at step {self.state.global_step}") from exc
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
    parser.add_argument("--output_dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--model_dir", default=None, help="Override final best-model directory")
    parser.add_argument("--resume_from_checkpoint", default="auto", help="auto, none, or a checkpoint path")
    parser.add_argument("--smoke_test_steps", type=int, default=0, help="Run a few training steps without saving a final model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model_dir:
        cfg["model_dir"] = args.model_dir
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
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    # Keep the previously documented conservative FP32 behaviour for mT5/ByT5.
    use_bf16 = (cuda and cfg["model_family"] == "nllb"
                and torch.cuda.get_device_capability(0)[0] >= 8
                and torch.cuda.is_bf16_supported())
    use_fp16 = cuda and cfg["model_family"] == "nllb" and not use_bf16
    print(f"Device: {gpu_name}; bf16: {use_bf16}; fp16: {use_fp16}")
    checkpointing = cfg.get("gradient_checkpointing", True)
    if checkpointing:
        model.config.use_cache = False
    checkpoint_dir = args.output_dir or cfg["checkpoint_dir"]
    interval = cfg.get("eval_save_steps", 1000)
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("eval_save_steps must be a positive integer")
    batch_size = cfg["batch_size"]
    accumulation = cfg["gradient_accumulation_steps"]
    # Trainer runs on one Colab GPU; warmup_steps avoids deprecated warmup_ratio.
    updates_per_epoch = max(1, math.ceil(len(train) / batch_size) // accumulation)
    warmup_steps = int(math.ceil(updates_per_epoch * cfg["epochs"] * cfg.get("warmup_ratio", 0.06)))
    if args.smoke_test_steps < 0:
        parser.error("--smoke_test_steps must be nonnegative")
    if args.smoke_test_steps and args.resume_from_checkpoint != "none":
        parser.error("A smoke test requires --resume_from_checkpoint none")
    if args.smoke_test_steps >= interval:
        parser.error("Smoke-test steps must be smaller than eval_save_steps")
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=interval,
        save_steps=interval,
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        seed=cfg["seed"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrf",
        greater_is_better=True,
        save_total_limit=2,
        max_steps=args.smoke_test_steps if args.smoke_test_steps else -1,
        report_to="none",
        logging_steps=50,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None,
        group_by_length=True,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        label_smoothing_factor=cfg.get("label_smoothing_factor", 0.1),
        weight_decay=cfg.get("weight_decay", 0.01),
        optim=cfg.get("optim", "adamw_torch"),
    )
    trainer = MultilingualTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=EncodedDataset(train, tokenizer, cfg),
        eval_dataset=EncodedDataset(validation, tokenizer, cfg),
        data_collator=TranslationCollator(tokenizer, model=model),
        validation_rows=validation,
        pipeline_config=cfg,
        drive_backup_dir=args.drive_backup_dir or cfg.get("drive_backup_dir"),
    )
    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = get_last_checkpoint(checkpoint_dir) if Path(checkpoint_dir).is_dir() else None
    elif resume.lower() == "none":
        resume = None
    print(f"Checkpoint directory: {checkpoint_dir}; resume: {resume or 'new run'}")
    trainer.train(resume_from_checkpoint=resume)
    if args.smoke_test_steps:
        sample = generate(validation[:2], trainer.model, tokenizer, cfg)
        if len(sample) != min(2, len(validation)) or any(not isinstance(p, str) for p in sample):
            raise RuntimeError("Smoke test generation did not return the expected strings")
        print("Smoke test training and generation finished; no final model saved.")
        return
    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg["model_dir"])
    tokenizer.save_pretrained(cfg["model_dir"])
    with (Path(cfg["model_dir"]) / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Best model saved to {cfg['model_dir']}")


if __name__ == "__main__":
    main()
