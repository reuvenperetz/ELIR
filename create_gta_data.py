import os
from pathlib import Path
import numpy as np
from PIL import Image


SRC_ROOT = Path("val/img")
DST_X2 = Path("val_x2/img")
DST_X4 = Path("val_x4/img")


def block_average_downsample(img, factor):
    """
    img: PIL Image (H, W, 3)
    factor: 2 or 4
    """
    arr = np.asarray(img, dtype=np.float32)

    h, w, c = arr.shape
    h2 = h // factor * factor
    w2 = w // factor * factor

    arr = arr[:h2, :w2]

    arr = arr.reshape(
        h2 // factor, factor,
        w2 // factor, factor,
        c
    ).mean(axis=(1, 3))

    return Image.fromarray(arr.astype(np.uint8))


def process_scale(dst_root, factor):
    print(f"Creating x{factor} dataset")

    for scene_dir in sorted(SRC_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene = scene_dir.name
        print(f"  Scene {scene}")

        hq_dir = dst_root / scene / "hq"
        lq_dir = dst_root / scene / "lq"
        hq_dir.mkdir(parents=True, exist_ok=True)
        lq_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(scene_dir.glob("*.png")):
            img = Image.open(img_path).convert("RGB")

            # HQ: original
            img.save(hq_dir / img_path.name)

            # LQ: block-averaged downsample
            lq = block_average_downsample(img, factor)
            lq.save(lq_dir / img_path.name)


if __name__ == "__main__":
    process_scale(DST_X2, factor=2)
    process_scale(DST_X4, factor=4)
    print("Done.")
