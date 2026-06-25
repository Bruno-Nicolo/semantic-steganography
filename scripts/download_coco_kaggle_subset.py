#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a COCO 2017 subset from Kaggle and split it into shards."
    )
    parser.add_argument("--dataset", default="awsaf49/coco-2017-dataset")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("data/coco/val2017_shards"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/.kaggle_coco_download"))
    parser.add_argument("--force", action="store_true", help="Overwrite output/work directories if they exist.")
    parser.add_argument("--quiet", action="store_true", help="Reduce Kaggle download output.")
    return parser.parse_args()


def load_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "Missing Kaggle API package. Install it with: pip install kaggle\n"
            "Then configure credentials in ~/.kaggle/kaggle.json or via env vars."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def prepare_directory(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Path already exists: {path}. Re-run with --force to overwrite it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_image_path(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in IMAGE_EXTENSIONS


def looks_like_split_path(path: str, split: str) -> bool:
    return split in PurePosixPath(path).parts


def select_items(items: list[str], count: int, seed: int) -> list[str]:
    if len(items) < count:
        raise SystemExit(f"Requested {count} images, but only found {len(items)} candidates.")
    rng = random.Random(seed)
    selected = rng.sample(sorted(items), count)
    return sorted(selected)


def shard_for_index(index: int, count: int, shards: int) -> int:
    return index * shards // count


def shard_dirs(output_root: Path, shards: int) -> list[Path]:
    dirs = [output_root / f"shard_{index:02d}" for index in range(shards)]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_manifest(
    output_root: Path,
    dataset: str,
    split: str,
    count: int,
    shards: int,
    seed: int,
    selected: list[str],
    source_mode: str,
) -> None:
    per_shard: dict[str, list[str]] = {f"shard_{index:02d}": [] for index in range(shards)}
    for index, item in enumerate(selected):
        per_shard[f"shard_{shard_for_index(index, count, shards):02d}"].append(Path(item).name)

    manifest = {
        "dataset": dataset,
        "split": split,
        "count": count,
        "shards": shards,
        "seed": seed,
        "source_mode": source_mode,
        "per_shard_counts": {name: len(files) for name, files in per_shard.items()},
        "per_shard_files": per_shard,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def dataset_file_names(api, dataset: str) -> list[str]:
    listing = api.dataset_list_files(dataset)
    return sorted(file.name for file in listing.files)


def download_individual_images(
    api,
    dataset: str,
    selected: list[str],
    output_root: Path,
    work_dir: Path,
    shards: int,
    quiet: bool,
) -> None:
    dirs = shard_dirs(output_root, shards)
    for index, kaggle_path in enumerate(selected):
        api.dataset_download_file(dataset, kaggle_path, path=str(work_dir), force=False, quiet=quiet)
        downloaded = work_dir / Path(kaggle_path).name
        if not downloaded.exists():
            matches = list(work_dir.rglob(Path(kaggle_path).name))
            if not matches:
                raise SystemExit(f"Kaggle download completed but file was not found: {kaggle_path}")
            downloaded = matches[0]

        destination = dirs[shard_for_index(index, len(selected), shards)] / downloaded.name
        shutil.copy2(downloaded, destination)


def choose_archive(file_names: list[str], split: str) -> str:
    archives = [name for name in file_names if PurePosixPath(name).suffix.lower() == ".zip"]
    if not archives:
        raise SystemExit("No individual images and no .zip archive found in the Kaggle dataset listing.")

    preferred = [name for name in archives if split in name or "coco2017" in name.lower()]
    return sorted(preferred or archives)[0]


def download_dataset_archive(api, dataset: str, work_dir: Path, force: bool, quiet: bool) -> Path:
    api.dataset_download_files(dataset, path=str(work_dir), unzip=False, force=force, quiet=quiet)
    archives = sorted(work_dir.glob("*.zip"))
    if not archives:
        raise SystemExit(f"Dataset download completed but no .zip archive was found in: {work_dir}")
    return archives[0]


def extract_selected_from_archive(
    api,
    dataset: str,
    archive_name: str,
    split: str,
    count: int,
    seed: int,
    output_root: Path,
    work_dir: Path,
    shards: int,
    force: bool,
    quiet: bool,
) -> list[str]:
    print(
        "Kaggle exposes this dataset as an archive. Downloading the archive, "
        "then extracting only the requested images.",
        file=sys.stderr,
    )
    api.dataset_download_file(dataset, archive_name, path=str(work_dir), force=force, quiet=quiet)
    archive_path = work_dir / Path(archive_name).name
    if not archive_path.exists():
        matches = list(work_dir.rglob(Path(archive_name).name))
        if not matches:
            raise SystemExit(f"Archive download completed but file was not found: {archive_name}")
        archive_path = matches[0]

    dirs = shard_dirs(output_root, shards)
    with zipfile.ZipFile(archive_path) as zip_file:
        image_members = [
            member
            for member in zip_file.namelist()
            if is_image_path(member) and looks_like_split_path(member, split)
        ]
        if not image_members:
            image_members = [member for member in zip_file.namelist() if is_image_path(member)]

        selected = select_items(image_members, count, seed)
        for index, member in enumerate(selected):
            destination = dirs[shard_for_index(index, count, shards)] / Path(member).name
            with zip_file.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    return selected


def extract_selected_from_archive_path(
    archive_path: Path,
    dataset: str,
    split: str,
    count: int,
    seed: int,
    output_root: Path,
    shards: int,
) -> list[str]:
    dirs = shard_dirs(output_root, shards)
    with zipfile.ZipFile(archive_path) as zip_file:
        image_members = [
            member
            for member in zip_file.namelist()
            if is_image_path(member) and looks_like_split_path(member, split)
        ]
        if not image_members:
            image_members = [member for member in zip_file.namelist() if is_image_path(member)]

        selected = select_items(image_members, count, seed)
        for index, member in enumerate(selected):
            destination = dirs[shard_for_index(index, count, shards)] / Path(member).name
            with zip_file.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    return selected


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be greater than 0.")
    if args.shards <= 0:
        raise SystemExit("--shards must be greater than 0.")

    prepare_directory(args.output_root, args.force)
    prepare_directory(args.work_dir, args.force)

    api = load_kaggle_api()
    file_names = dataset_file_names(api, args.dataset)
    image_files = [name for name in file_names if is_image_path(name) and looks_like_split_path(name, args.split)]

    if len(image_files) >= args.count:
        selected = select_items(image_files, args.count, args.seed)
        download_individual_images(
            api, args.dataset, selected, args.output_root, args.work_dir, args.shards, args.quiet
        )
        source_mode = "individual_files"
    else:
        try:
            archive_name = choose_archive(file_names, args.split)
        except SystemExit:
            print(
                "Kaggle did not expose individual files in the listing. "
                "Downloading the dataset archive instead.",
                file=sys.stderr,
            )
            archive_path = download_dataset_archive(api, args.dataset, args.work_dir, args.force, args.quiet)
            selected = extract_selected_from_archive_path(
                archive_path,
                args.dataset,
                args.split,
                args.count,
                args.seed,
                args.output_root,
                args.shards,
            )
            source_mode = f"dataset_archive:{archive_path.name}"
        else:
            selected = extract_selected_from_archive(
                api,
                args.dataset,
                archive_name,
                args.split,
                args.count,
                args.seed,
                args.output_root,
                args.work_dir,
                args.shards,
                args.force,
                args.quiet,
            )
            source_mode = f"archive:{archive_name}"

    write_manifest(
        args.output_root,
        args.dataset,
        args.split,
        args.count,
        args.shards,
        args.seed,
        selected,
        source_mode,
    )
    print(f"Wrote {args.count} images into {args.shards} shards under {args.output_root}")


if __name__ == "__main__":
    main()
