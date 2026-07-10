"""Dataset loading helpers for ConvFinQA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, NamedTuple

from src.models import ConvFinQADataset, ConvFinQARecord

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "convfinqa_dataset.json"
DatasetSplit = Literal["train", "dev"]


class LocatedRecord(NamedTuple):
    """A record plus the split it was loaded from."""

    split: DatasetSplit
    record: ConvFinQARecord


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> ConvFinQADataset:
    """Load and validate the cleaned ConvFinQA dataset."""
    with path.open() as file:
        raw_data = json.load(file)

    return ConvFinQADataset.model_validate(raw_data)


def find_record(dataset: ConvFinQADataset, record_id: str) -> LocatedRecord | None:
    """Find a record by ID across train and dev splits."""
    for split in ("train", "dev"):
        records = getattr(dataset, split)
        for record in records:
            if record.id == record_id:
                return LocatedRecord(split=split, record=record)

    return None
