
from collections import defaultdict


class PatchTracker:
    def __init__(self):
        self.loaded_patches = defaultdict(set)  # {image_idx: set of (y, x) tuples}
        self.expected_patches = {}  # {image_idx: set of (y, x) tuples}

    def register_image(self, image_idx, height, width, patch_size=256, stride=256):
        """Register expected patches for an image."""
        expected = set()
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                expected.add((y, x))
        # Handle edge patches if image doesn't divide evenly
        if (height - patch_size) % stride != 0:
            y = height - patch_size
            for x in range(0, width - patch_size + 1, stride):
                expected.add((y, x))
        if (width - patch_size) % stride != 0:
            x = width - patch_size
            for y in range(0, height - patch_size + 1, stride):
                expected.add((y, x))
        if (height - patch_size) % stride != 0 and (width - patch_size) % stride != 0:
            expected.add((height - patch_size, width - patch_size))

        self.expected_patches[image_idx] = expected
        # print(f"Image {image_idx}: expecting {len(expected)} patches")

    def log_patch(self, patch_idx, image_idx, y, x):
        """Call this when loading a patch."""
        self.loaded_patches[image_idx].add((y, x))
        # Uncomment for verbose logging:
        # print(f"Loading patch {patch_idx}: Image {image_idx}, y={y}, x={x}")

    def verify(self):
        """Check if all expected patches were loaded."""
        all_good = True
        for img_idx, expected in self.expected_patches.items():
            loaded = self.loaded_patches[img_idx]
            missing = expected - loaded
            extra = loaded - expected

            print(f"\n=== Image {img_idx} ===")
            print(f"Expected: {len(expected)}, Loaded: {len(loaded)}")

            if missing:
                print(f"❌ MISSING {len(missing)} patches:")
                for y, x in sorted(missing)[:10]:  # Show first 10
                    print(f"   y={y}, x={x}")
                if len(missing) > 10:
                    print(f"   ... and {len(missing) - 10} more")
                all_good = False

            if extra:
                print(f"⚠️  EXTRA {len(extra)} patches (unexpected coordinates)")
                all_good = False

            if not missing and not extra:
                print(f"✅ All patches accounted for!")

        return all_good

