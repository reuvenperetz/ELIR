import os
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_list_file(list_path, is_physics_format=False):
    """
    Parse SID list file.
    Original format: ./Sony/short/10003_00_0.1s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F8
    Physics format: 10003_00_0.1s 10003_00_10s 100 (filenames + ratio)
    """
    pairs = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            if is_physics_format:
                short_name = parts[0]
                long_name = parts[1]
                ratio = int(parts[2]) if len(parts) > 2 else None

                if not short_name.endswith(".ARW"):
                    short_name += ".ARW"
                if not long_name.endswith(".ARW"):
                    long_name += ".ARW"
                short_path = f"Sony/short/{short_name}"
                long_path = f"Sony/long/{long_name}"
                pairs.append((short_path, long_path, ratio))
            else:
                short_path = parts[0].lstrip("./")
                long_path = parts[1].lstrip("./")
                pairs.append((short_path, long_path, None))

    return pairs

def arrange_sid_dataset(archive_path, output_path, list_dir, no_val=False, physics_split=None):
    """
    Arrange SID Sony dataset into lq/hq folder pairs based on train/val/test lists.

    Args:
        archive_path: Path to the 'archive' folder containing Sony/long and Sony/short
        output_path: Path where train/val/test folders with lq/hq subfolders will be created
        list_dir: Directory containing Sony_train_list.txt, Sony_val_list.txt, Sony_test_list.txt
        no_val: If True, merge validation into training (only train/test splits)
        physics_split: Path to SID_Sony_15_paired.txt file for ELD/physics-based partition.
                      If provided, creates train/test split where test contains pairs from this file
                      and train contains all other pairs.
    """
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    list_dir = Path(list_dir)

    if physics_split:
        physics_split = Path(physics_split)
        if not physics_split.exists():
            raise FileNotFoundError(f"Physics split file not found: {physics_split}")

        print(f"Using physics-based partition from: {physics_split}")

        # Load test pairs from physics split file (with ratios)
        test_pairs_with_ratio = parse_list_file(physics_split, is_physics_format=True)
        test_pairs_set = set((p[0], p[1]) for p in test_pairs_with_ratio)
        print(f"Loaded {len(test_pairs_with_ratio)} test pairs from physics split file")

        # Group test pairs by ratio
        from collections import defaultdict
        test_by_ratio = defaultdict(list)
        for short_path, long_path, ratio in test_pairs_with_ratio:
            test_by_ratio[ratio].append((short_path, long_path))

        print(f"Test pairs by ratio: {dict((k, len(v)) for k, v in test_by_ratio.items())}")

        # Load all pairs from original partition files
        all_pairs = []
        for list_file in ["Sony_train_list.txt", "Sony_val_list.txt", "Sony_test_list.txt"]:
            list_path = list_dir / list_file
            if list_path.exists():
                pairs = parse_list_file(list_path, is_physics_format=False)
                all_pairs.extend(pairs)
                print(f"  Loaded {len(pairs)} pairs from {list_file}")

        # Remove duplicates
        seen = set()
        unique_pairs = []
        for short_path, long_path, _ in all_pairs:
            if (short_path, long_path) not in seen:
                seen.add((short_path, long_path))
                unique_pairs.append((short_path, long_path))
        print(f"Total unique pairs: {len(unique_pairs)}")

        # Train is everything not in test
        train_pairs = [(s, l) for s, l in unique_pairs if (s, l) not in test_pairs_set]

        print(f"\nPhysics partition:")
        print(f"  Train: {len(train_pairs)} pairs")
        for ratio, pairs in sorted(test_by_ratio.items()):
            print(f"  Test x{ratio}: {len(pairs)} pairs")

        # Build splits dict: train + test_x100, test_x250, test_x300
        splits = {"train": train_pairs}
        for ratio, pairs in test_by_ratio.items():
            splits[f"test_x{ratio}"] = pairs

        # Process each split
        for split_name, pairs in splits.items():
            print(f"\n{'=' * 50}")
            print(f"Processing {split_name} split...")
            print(f"{'=' * 50}")

            lq_dir = output_path / split_name / "lq"
            hq_dir = output_path / split_name / "hq"
            lq_dir.mkdir(parents=True, exist_ok=True)
            hq_dir.mkdir(parents=True, exist_ok=True)

            missing = []
            copied = 0
            for idx, (short_rel, long_rel) in enumerate(tqdm(pairs, desc=f"Arranging {split_name}")):
                short_path = archive_path / short_rel
                long_path = archive_path / long_rel

                if not short_path.exists():
                    missing.append(f"short: {short_rel}")
                    continue
                if not long_path.exists():
                    missing.append(f"long: {long_rel}")
                    continue

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
    else:
        # Original SID partition
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
    parser.add_argument("--physics-split", type=str, default=None,
                        help="Path to SID_Sony_15_paired.txt for ELD/physics-based partition. "
                             "Test set will contain pairs from this file, train will contain all others.")

    args = parser.parse_args()

    arrange_sid_dataset(args.archive_path, args.output, args.list_dir, args.no_val, args.physics_split)