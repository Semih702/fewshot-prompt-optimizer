from __future__ import annotations

import json
from pathlib import Path

from prompt_from_examples.models import SupportDataset


def load_dataset(path: str | Path) -> SupportDataset:
    dataset_path = Path(path)
    raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return SupportDataset.model_validate(raw_data)
