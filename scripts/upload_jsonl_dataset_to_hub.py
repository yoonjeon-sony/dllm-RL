#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import create_repo


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
IMAGE_COLUMN_PATTERNS = (
    re.compile(r"^image\d+$", re.IGNORECASE),
    re.compile(r"^problem_image(?:_\d+)?$", re.IGNORECASE),
    re.compile(r"^reasoning_image(?:_\d+)?$", re.IGNORECASE),
    re.compile(r"^visdiff$", re.IGNORECASE),
)
SPLIT_SUFFIXES = (
    ("_train", "train"),
    ("_val", "validation"),
    ("-train", "train"),
    ("-val", "validation"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local JSONL datasets to the Hugging Face Hub with image columns cast as datasets.Image()."
    )
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face dataset repo, e.g. username/repo-name")
    parser.add_argument("--data-dir", type=Path, default=Path("train_sft/data"), help="Directory containing JSONL files")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/group2/dgm/yoonjeon/data"),
        help="Root directory that contains the source image folders",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only upload files whose basename contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip files whose basename contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Only process the first N matching JSONL files. Useful for testing.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1000,
        help="How many rows to scan when detecting image columns and validating paths.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="If set, truncate each JSONL-derived split to at most this many rows before upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate records and print the upload plan without pushing to the Hub.",
    )
    return parser.parse_args()


def matches_filters(path: Path, include_filters: list[str], exclude_filters: list[str]) -> bool:
    name = path.name
    if include_filters and not any(token in name for token in include_filters):
        return False
    if any(token in name for token in exclude_filters):
        return False
    return True


def infer_config_and_split(stem: str) -> tuple[str, str]:
    for suffix, split in SPLIT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    return stem, "train"


def sanitize_config_name(name: str) -> str:
    name = name.strip().replace("&", "and")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z._-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def build_candidate_image_dirs(file_stem: str) -> list[str]:
    candidates = [file_stem]
    for suffix, _ in SPLIT_SUFFIXES:
        if file_stem.endswith(suffix):
            candidates.append(file_stem[: -len(suffix)])
    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.endswith("_loc"):
            expanded.append(candidate[: -len("_loc")])
    deduped = []
    seen = set()
    for candidate in expanded:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def is_image_reference(value: Any) -> bool:
    return isinstance(value, str) and Path(value.strip()).suffix.lower() in IMAGE_EXTENSIONS


def looks_like_image_column(column_name: str) -> bool:
    return any(pattern.match(column_name) for pattern in IMAGE_COLUMN_PATTERNS)


def detect_image_columns(jsonl_path: Path, sample_rows: int) -> list[str]:
    columns = set()
    with jsonl_path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index >= sample_rows:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key, value in record.items():
                if looks_like_image_column(key) or is_image_reference(value):
                    columns.add(key)
    return sorted(columns)


def resolve_image_path(value: str, file_stem: str, image_root: Path) -> str:
    normalized = value.strip().replace("\\", "/")
    direct_path = Path(normalized).expanduser()
    if direct_path.is_file():
        return str(direct_path)

    relative_value = normalized[2:] if normalized.startswith("./") else normalized
    rooted_path = image_root / relative_value
    if rooted_path.is_file():
        return str(rooted_path)

    image_name = Path(relative_value).name
    for candidate_dir in build_candidate_image_dirs(file_stem):
        candidate_path = image_root / candidate_dir / image_name
        if candidate_path.is_file():
            return str(candidate_path)

    raise FileNotFoundError(
        f"Could not resolve image path '{value}' for dataset file '{file_stem}'. Looked under '{image_root}'."
    )


def validate_non_empty(jsonl_path: Path) -> bool:
    return jsonl_path.stat().st_size > 0


def validate_image_paths(jsonl_path: Path, image_columns: list[str], image_root: Path, sample_rows: int) -> None:
    if not image_columns:
        return

    with jsonl_path.open() as handle:
        checked = 0
        for line in handle:
            if checked >= sample_rows:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for column in image_columns:
                value = record.get(column)
                if is_image_reference(value):
                    resolve_image_path(value, jsonl_path.stem, image_root)
            checked += 1


def load_json_dataset(jsonl_path: Path) -> datasets.Dataset:
    rows: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    with jsonl_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            normalized = {
                key: json.dumps(value, ensure_ascii=True, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in record.items()
            }
            rows.append(normalized)
            all_keys.update(normalized.keys())

    if not rows:
        return datasets.Dataset.from_list([])

    normalized_rows = [{key: row.get(key) for key in all_keys} for row in rows]
    return datasets.Dataset.from_list(normalized_rows)


def truncate_dataset(dataset: datasets.Dataset, max_rows_per_file: int | None) -> datasets.Dataset:
    if max_rows_per_file is None or dataset.num_rows <= max_rows_per_file:
        return dataset
    return dataset.select(range(max_rows_per_file))


def cast_image_columns(
    dataset: datasets.Dataset, image_columns: list[str], file_stem: str, image_root: Path
) -> datasets.Dataset:
    if not image_columns:
        return dataset

    def mapper(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        updates: dict[str, list[Any]] = {}
        for column in image_columns:
            values = batch[column]
            updates[column] = [
                resolve_image_path(value, file_stem, image_root) if is_image_reference(value) else value for value in values
            ]
        return updates

    dataset = dataset.map(mapper, batched=True, desc=f"Resolving image paths for {file_stem}")
    for column in image_columns:
        dataset = dataset.cast_column(column, datasets.Image())
    return dataset


def group_jsonl_files(paths: list[Path]) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in paths:
        config_name, split_name = infer_config_and_split(path.stem)
        grouped.setdefault(sanitize_config_name(config_name), {})[split_name] = path
    return grouped


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    image_root = args.image_root.resolve()

    jsonl_paths = sorted(path for path in data_dir.glob("*.jsonl") if matches_filters(path, args.include, args.exclude))
    if args.limit_files is not None:
        jsonl_paths = jsonl_paths[: args.limit_files]

    if not jsonl_paths:
        raise SystemExit("No JSONL files matched the requested filters.")

    grouped_paths = group_jsonl_files(jsonl_paths)
    skipped_empty: list[Path] = []

    if not args.dry_run:
        create_repo(repo_id=args.repo_id, repo_type="dataset", token=args.token, private=args.private, exist_ok=True)

    for config_name, split_paths in grouped_paths.items():
        dataset_dict = datasets.DatasetDict()
        print(f"Preparing config '{config_name}'")

        for split_name, jsonl_path in sorted(split_paths.items()):
            if not validate_non_empty(jsonl_path):
                skipped_empty.append(jsonl_path)
                print(f"  - skipping empty file: {jsonl_path.name}")
                continue

            image_columns = detect_image_columns(jsonl_path, sample_rows=args.sample_rows)
            validate_image_paths(jsonl_path, image_columns, image_root, sample_rows=args.sample_rows)

            dataset = load_json_dataset(jsonl_path)
            dataset = truncate_dataset(dataset, args.max_rows_per_file)
            dataset = cast_image_columns(dataset, image_columns, jsonl_path.stem, image_root)
            dataset_dict[split_name] = dataset

            print(
                f"  - split={split_name} rows={dataset.num_rows} image_columns={image_columns or 'none'} source={jsonl_path.name}"
            )

        if not dataset_dict:
            print(f"  - no non-empty splits found for config '{config_name}', skipping")
            continue

        if args.dry_run:
            continue

        dataset_dict.push_to_hub(
            repo_id=args.repo_id,
            config_name=config_name,
            token=args.token,
        )
        print(f"  - pushed config '{config_name}' to {args.repo_id}")

    if skipped_empty:
        skipped_names = ", ".join(path.name for path in skipped_empty)
        print(f"Skipped empty files: {skipped_names}")


if __name__ == "__main__":
    main()
