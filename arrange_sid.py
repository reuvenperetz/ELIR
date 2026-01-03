import os
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_list_file(list_path):
    """
    Parse SID list file.
    Each line: ./Sony/short/10003_00_0.1s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F8
    """
    pairs = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            short_path = parts[0].lstrip("./")
            long_path = parts[1].lstrip("./")
            pairs.append((short_path, long_path))

    return pairs


def arrange_sid_dataset(archive_path, output_path, list_dir, no_val=False):
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    list_dir = Path(list_dir)

    if no_val:
        splits = {
            "train": ["Sony_train_list.txt", "Sony_val_list.txt"],
            "test": ["Sony_test_list.txt"],
        }
    else:
        splits = {
            "train": ["Sony_train_list.txt"],
            "val": ["Sony_val_list.txt"],
            "test": ["Sony_test_list.txt"],
        }

    for split_name, list_files in splits.items():
        print(f"\n{'=' * 50}")
        print(f"Processing {split_name} split...")
        print(f"{'=' * 50}")

        lq_dir = output_path / split_name / "lq"
        hq_dir = output_path / split_name / "hq"
        lq_dir.mkdir(parents=True, exist_ok=True)
        hq_dir.mkdir(parents=True, exist_ok=True)

        all_pairs = []
        for list_file in list_files:
            list_path = list_dir / list_file
            if not list_path.exists():
                print(f"⚠️  List file not found: {list_path}")
                continue
            pairs = parse_list_file(list_path)
            all_pairs.extend(pairs)
            print(f"  Loaded {len(pairs)} pairs from {list_file}")

        missing = []
        copied = 0
        for idx, (short_rel, long_rel) in enumerate(tqdm(all_pairs, desc=f"Arranging {split_name}")):
            short_path = archive_path / short_rel
            long_path = archive_path / long_rel

            if not short_path.exists():
                missing.append(f"short: {short_rel}")
                continue
            if not long_path.exists():
                missing.append(f"long: {long_rel}")
                continue

            # Use index to create unique paired filenames
            # This ensures each lq/hq pair has matching names
            ext_short = short_path.suffix
            ext_long = long_path.suffix

            shutil.copy2(short_path, lq_dir / f"{idx:05d}{ext_short}")
            shutil.copy2(long_path, hq_dir / f"{idx:05d}{ext_long}")
            copied += 1

        if missing:
            print(f"\n⚠️  {len(missing)} files missing:")
            for m in missing[:5]:
                print(f"   {m}")
            if len(missing) > 5:
                print(f"   ... and {len(missing) - 5} more")

        print(f"✅ {split_name}: {copied} pairs")

    print(f"\n{'=' * 50}")
    print("Done!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arrange SID dataset into lq/hq folders")
    parser.add_argument("archive_path", help="Path to 'archive' folder")
    parser.add_argument("list_dir", help="Directory containing Sony_*_list.txt files")
    parser.add_argument("--output", "-o", default="./dataset", help="Output path (default: ./dataset)")
    parser.add_argument("--no-val", action="store_true", help="Merge validation into training (only train/test)")

    args = parser.parse_args()

    arrange_sid_dataset(args.archive_path, args.output, args.list_dir, args.no_val)
