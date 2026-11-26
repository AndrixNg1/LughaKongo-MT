"""Shared data validation, language conditioning and generation."""

import csv
import json
from collections import defaultdict
from pathlib import Path

LANGS = {"fr", "shr", "ngc", "yns", "tap", "hav"}
TAGS = {"fr": "fra_Latn", **{lang: f"{lang}_Latn" for lang in LANGS - {"fr"}}}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if config["model_family"] not in {"nllb", "t5"}:
        raise ValueError("model_family must be 'nllb' or 't5'")
    return config


def read_rows(path, labeled=False):
    required = {"ID", "SOURCE_LANG", "TARGET_LANG", "SOURCE_TEXT"}
    if labeled:
        required |= {"TARGET_TEXT", "VERSE_KEY"}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing columns in {path}: {required - set(reader.fieldnames or [])}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty file: {path}")
    ids = set()
    for row in rows:
        src, tgt = row["SOURCE_LANG"], row["TARGET_LANG"]
        if src not in LANGS or tgt not in LANGS or ((src == "fr") + (tgt == "fr")) != 1:
            raise ValueError(f"Invalid direction: {src} -> {tgt}")
        for field in required:
            if not (row.get(field) or "").strip():
                raise ValueError(f"Empty {field} in {path}")
        if row["ID"] in ids:
            raise ValueError(f"Duplicate ID in {path}: {row['ID']}")
        ids.add(row["ID"])
    return rows


def initialize_tokenizer(tokenizer, model, family):
    if family == "nllb":
        tokens = [TAGS[lang] for lang in sorted(LANGS - {"fr"})]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens},
            replace_additional_special_tokens=False,
        )
        model.resize_token_embeddings(len(tokenizer))
        for tag in TAGS.values():
            if tokenizer.convert_tokens_to_ids(tag) == tokenizer.unk_token_id:
                raise ValueError(f"Missing language token: {tag}")


def encode(row, tokenizer, config, labeled=True):
    src, tgt = row["SOURCE_LANG"], row["TARGET_LANG"]
    if config["model_family"] == "nllb":
        # Explicit language tokens avoid tokenizer state changes in mixed-direction batches.
        source = tokenizer.encode(
            row["SOURCE_TEXT"],
            add_special_tokens=False,
            truncation=True,
            max_length=config["max_source_length"] - 2,
        )
        result = {
            "input_ids": [tokenizer.convert_tokens_to_ids(TAGS[src])]
            + source
            + [tokenizer.eos_token_id]
        }
        if labeled:
            target = tokenizer.encode(
                row["TARGET_TEXT"],
                add_special_tokens=False,
                truncation=True,
                max_length=config["max_target_length"] - 2,
            )
            result["labels"] = (
                [tokenizer.convert_tokens_to_ids(TAGS[tgt])]
                + target
                + [tokenizer.eos_token_id]
            )
    else:
        prefix = f"translate {src} to {tgt}: "
        result = tokenizer(
            prefix + row["SOURCE_TEXT"],
            truncation=True,
            max_length=config["max_source_length"],
        )
        if labeled:
            result["labels"] = tokenizer(
                text_target=row["TARGET_TEXT"],
                truncation=True,
                max_length=config["max_target_length"],
            )["input_ids"]
    result["attention_mask"] = [1] * len(result["input_ids"])
    return result


def generate(rows, model, tokenizer, config):
    import torch

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["TARGET_LANG"]].append((i, row))
    predictions = [None] * len(rows)
    model.eval()
    for target, group in groups.items():
        for start in range(0, len(group), config["batch_size"]):
            chunk = group[start : start + config["batch_size"]]
            encoded = [
                encode(row, tokenizer, config, labeled=False) for _, row in chunk
            ]
            batch = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(
                model.device
            )
            kwargs = {
                "max_new_tokens": config["max_target_length"],
                "num_beams": config["num_beams"],
            }
            if config["model_family"] == "nllb":
                kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(
                    TAGS[target]
                )
            with torch.inference_mode():
                output = model.generate(**batch, **kwargs)
            for (index, _), prediction in zip(
                chunk, tokenizer.batch_decode(output, skip_special_tokens=True)
            ):
                predictions[index] = prediction.strip()
    return predictions
