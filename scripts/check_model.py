"""Inspect pinned local Qwen files without loading a model, network, or robot."""

import argparse
import hashlib
import json
from pathlib import Path


WEIGHTS = {
    "model-00001-of-00002.safetensors": (
        4967229296,
        "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    ),
    "model-00002-of-00002.safetensors": (
        3908490048,
        "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
    ),
}
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def inspect(root, verify_sha256=False):
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("Expected qwen3_vl model configuration")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(WEIGHTS):
        raise ValueError("Weight index does not match the pinned two-shard model")
    for name in ("tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        json.loads((root / name).read_text())
    results = {}
    for name, (size, expected_hash) in WEIGHTS.items():
        path = root / name
        if path.stat().st_size != size:
            raise ValueError(f"Incomplete or unexpected weights: {name}")
        row = {"bytes": size, "size_ok": True, "sha256_verified": False}
        if verify_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"SHA-256 mismatch: {name}")
            row.update(sha256=digest.hexdigest(), sha256_verified=True)
        results[name] = row
    return {
        "model_dir": str(root),
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "expected_revision": REVISION,
        "files": results,
        "model_loaded": False,
        "training_tested": False,
        "robot_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    try:
        report = inspect(args.model_dir.expanduser().resolve(), args.sha256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Model check failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
