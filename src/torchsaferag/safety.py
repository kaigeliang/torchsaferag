from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_attack_templates(path: Path) -> List[str]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_attacks(templates: List[str], base_questions: List[str], limit: int) -> List[Dict[str, Any]]:
    attacks = []
    for i, t in enumerate(templates):
        for q in base_questions:
            attacks.append({"id": f"attack_{i}", "question": f"{q}\n\n{t}"})
    random.shuffle(attacks)
    return attacks[:limit]


def load_poison_chunks(poison_dir: Path) -> List[Dict[str, Any]]:
    chunks = []
    for path in poison_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks
