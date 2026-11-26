"""Generate and validate a submission aligned to the sample's IDs."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from common import generate, load_config, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_nllb.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = read_rows(cfg["test_file"])
    with open(cfg["sample_file"], encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["ID", "TARGET"]:
            raise ValueError("Sample must have exactly ID,TARGET columns")
        sample = list(reader)
    ids = [r["ID"] for r in sample]
    if not ids or any(not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Sample IDs must be nonempty and unique")
    if set(ids) != {r["ID"] for r in rows}:
        raise ValueError("Test and sample IDs differ")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_dir"])
    import torch

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    predictions = generate(rows, model, tokenizer, cfg)
    mapping = {row["ID"]: text for row, text in zip(rows, predictions)}
    if len(mapping) != len(ids) or any(
        not isinstance(mapping[i], str) or not mapping[i].strip() for i in ids
    ):
        raise ValueError("Missing or empty predictions; submission not written")
    output = Path(cfg["submission_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=output.parent, delete=False
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["ID", "TARGET"])
            writer.writerows((i, mapping[i]) for i in ids)
        with temporary.open(encoding="utf-8", newline="") as f:
            check = list(csv.DictReader(f))
        if [r["ID"] for r in check] != ids or any(
            not r["TARGET"].strip() for r in check
        ):
            raise ValueError("Final CSV verification failed")
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Saved {len(ids)} translations to {output}")


if __name__ == "__main__":
    main()
