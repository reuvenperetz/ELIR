import os
import shutil
import json
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def extract_exposure_time(filename):
    """
    Extract exposure time from SID filename.
    E.g., '10003_00_0.1s.ARW' -> 0.1
          '10003_00_10s.ARW' -> 10.0
    """
    # Match patterns like 0.1s, 10s, 0.04s, etc.
    match = re.search(r'_(\d+(?:\.\d+)?)s\.ARW$', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def compute_amplification_ratio(short_path, long_path):
    """
    Compute amplification ratio from exposure times in filenames.
    Ratio = long_exposure / short_exposure
    """
    short_filename = os.path.basename(short_path)
    long_filename = os.path.basename(long_path)

    short_exp = extract_exposure_time(short_filename)
    long_exp = extract_exposure_time(long_filename)

    if short_exp is not None and long_exp is not None and short_exp > 0:
        return long_exp / short_exp
    return None


def parse_list_file(list_path, is_physics_format=False):
    """
    Parse SID list file.
    Original format: ./Sony/short/10003_00_0.1s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F8
    Physics format: 10003_00_0.1s 10003_00_10s 100 (filenames + ratio)

    Returns list of tuples: (short_path, long_path, amplification_ratio)
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
                ratio = float(parts[2]) if len(parts) > 2 else None

                if not short_name.endswith(".ARW"):
                    short_name += ".ARW"
                if not long_name.endswith(".ARW"):
                    long_name += ".ARW"
                short_path = f"Sony/short/{short_name}"
                long_path = f"Sony/long/{long_name}"

                # If ratio not provided, compute from exposure times
                if ratio is None:
                    ratio = compute_amplification_ratio(short_path, long_path)

                pairs.append((short_path, long_path, ratio))
            else:
                short_path = parts[0].lstrip("./")
                long_path = parts[1].lstrip("./")
                # Compute ratio from exposure times in filenames
                ratio = compute_amplification_ratio(short_path, long_path)
                pairs.append((short_path, long_path, ratio))

    return pairs


def arrange_sid_dataset_dedup(archive_path, output_path, list_dir, no_val=False, physics_split=None):
    """
    Arrange SID Sony dataset with deduplicated HQ images.

    Instead of copying the same HQ image multiple times, we:
    1. Copy each unique HQ image only once
    2. Copy all LQ images
    3. Create a mapping.json file that maps LQ filenames to HQ filenames

    This saves significant disk space since each HQ image is paired with multiple LQ images.

    Args:
        archive_path: Path to the 'archive' folder containing Sony/long and Sony/short
        output_path: Path where train/val/test folders with lq/hq subfolders will be created
        list_dir: Directory containing Sony_train_list.txt, Sony_val_list.txt, Sony_test_list.txt
        no_val: If True, merge validation into training (only train/test splits)
        physics_split: Path to SID_Sony_15_paired.txt file for ELD/physics-based partition.
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

        # Remove duplicates (keep first occurrence with its ratio)
        seen = set()
        unique_pairs = []
        pair_to_ratio = {}  # Map (short, long) -> ratio
        for short_path, long_path, ratio in all_pairs:
            if (short_path, long_path) not in seen:
                seen.add((short_path, long_path))
                unique_pairs.append((short_path, long_path, ratio))
                pair_to_ratio[(short_path, long_path)] = ratio
        print(f"Total unique pairs: {len(unique_pairs)}")

        # Train is everything not in test (preserve ratios)
        train_pairs = [(s, l, r) for s, l, r in unique_pairs if (s, l) not in test_pairs_set]

        print(f"\nPhysics partition:")
        print(f"  Train: {len(train_pairs)} pairs")
        for ratio, pairs in sorted(test_by_ratio.items()):
            print(f"  Test x{ratio}: {len(pairs)} pairs")

        # Build splits dict: train + test_x100, test_x250, test_x300
        # All entries are now (short_path, long_path, ratio) tuples
        splits = {"train": train_pairs}
        for ratio, pairs in test_by_ratio.items():
            # Add ratio to each pair in test splits
            splits[f"test_x{int(ratio)}"] = [(s, l, ratio) for s, l in pairs]

    else:
        # Original SID partition
        if no_val:
            split_files = {
                "train": ["Sony_train_list.txt", "Sony_val_list.txt"],
                "test": ["Sony_test_list.txt"],
            }
        else:
            split_files = {
                "train": ["Sony_train_list.txt"],
                "val": ["Sony_val_list.txt"],
                "test": ["Sony_test_list.txt"],
            }

        splits = {}
        for split_name, list_files in split_files.items():
            all_pairs = []
            for list_file in list_files:
                list_path = list_dir / list_file
                if not list_path.exists():
                    print(f"⚠️  List file not found: {list_path}")
                    continue
                pairs = parse_list_file(list_path)
                # Keep full tuple (short, long, ratio)
                all_pairs.extend(pairs)
                print(f"  Loaded {len(pairs)} pairs from {list_file}")
            splits[split_name] = all_pairs

    # Process each split with deduplication
    for split_name, pairs in splits.items():
        print(f"\n{'=' * 50}")
        print(f"Processing {split_name} split (with HQ deduplication)...")
        print(f"{'=' * 50}")

        lq_dir = output_path / split_name / "lq"
        hq_dir = output_path / split_name / "hq"
        lq_dir.mkdir(parents=True, exist_ok=True)
        hq_dir.mkdir(parents=True, exist_ok=True)

        # Group LQ images by their HQ counterpart, preserving ratio info
        # hq_to_lq[long_rel] = [(short_rel, ratio), ...]
        hq_to_lq = defaultdict(list)
        for short_rel, long_rel, ratio in pairs:
            hq_to_lq[long_rel].append((short_rel, ratio))

        print(f"  Total pairs: {len(pairs)}")
        print(f"  Unique HQ images: {len(hq_to_lq)}")
        print(f"  Space savings: {len(pairs) - len(hq_to_lq)} fewer HQ copies")

        # Mapping: lq_filename -> {"hq": hq_filename, "amplification_ratio": ratio}
        mapping = {}
        missing = []
        lq_copied = 0
        hq_copied = 0

        # Process each unique HQ image
        for hq_idx, (long_rel, short_list) in enumerate(tqdm(hq_to_lq.items(), desc=f"Arranging {split_name}")):
            long_path = archive_path / long_rel

            if not long_path.exists():
                missing.append(f"long: {long_rel}")
                continue

            # Copy HQ image once
            ext_long = long_path.suffix
            hq_filename = f"{hq_idx:05d}{ext_long}"
            shutil.copy2(long_path, hq_dir / hq_filename)
            hq_copied += 1

            # Copy all corresponding LQ images
            for lq_idx, (short_rel, ratio) in enumerate(short_list):
                short_path = archive_path / short_rel

                if not short_path.exists():
                    missing.append(f"short: {short_rel}")
                    continue

                ext_short = short_path.suffix
                # Use combined index to ensure unique LQ filenames
                lq_filename = f"{hq_idx:05d}_{lq_idx:02d}{ext_short}"
                shutil.copy2(short_path, lq_dir / lq_filename)
                lq_copied += 1

                # Add to mapping with amplification ratio and original filenames for documentation
                mapping[lq_filename] = {
                    "hq": hq_filename,
                    "amplification_ratio": ratio,
                    "original_lq": os.path.basename(short_rel),
                    "original_hq": os.path.basename(long_rel)
                }

        # Save mapping file
        mapping_path = output_path / split_name / "mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)

        if missing:
            print(f"\n⚠️  {len(missing)} files missing:")
            for m in missing[:5]:
                print(f"   {m}")
            if len(missing) > 5:
                print(f"   ... and {len(missing) - 5} more")

        print(f"✅ {split_name}:")
        print(f"   LQ images: {lq_copied}")
        print(f"   HQ images: {hq_copied} (deduplicated from {len(pairs)} pairs)")
        print(f"   Mapping saved to: {mapping_path}")

    print(f"\n{'=' * 50}")
    print("Done!")
    print(f"{'=' * 50}")


# Keep the old function for backwards compatibility
def arrange_sid_dataset(archive_path, output_path, list_dir, no_val=False, physics_split=None):
    """
    Original function that duplicates HQ images.
    Use arrange_sid_dataset_dedup for space-efficient version.
    """
    # ... (original implementation kept for backwards compatibility)
    arrange_sid_dataset_dedup(archive_path, output_path, list_dir, no_val, physics_split)


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

    arrange_sid_dataset_dedup(args.archive_path, args.output, args.list_dir, args.no_val, args.physics_split)
