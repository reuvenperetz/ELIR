#!/bin/bash
set -e

# -------- paths --------
SRC_ROOT="QRISP/TestSet"
DST_ROOT="QRISP_FLAT_270_TO_1080"

LQ_DIR="${DST_ROOT}/lq"
HQ_DIR="${DST_ROOT}/hq"

# -------- create output dirs --------
mkdir -p "${LQ_DIR}"
mkdir -p "${HQ_DIR}"

# -------- iterate scenes --------
for scene in "${SRC_ROOT}"/*; do
    scene_name=$(basename "${scene}")

    echo "Processing scene: ${scene_name}"

    LQ_NATIVE="${scene}/270p/Native"
    HQ_NATIVE="${scene}/1080p/Native"

    # sanity check
    if [[ ! -d "${LQ_NATIVE}" || ! -d "${HQ_NATIVE}" ]]; then
        echo "Skipping ${scene_name} (missing Native folders)"
        continue
    fi

    # -------- iterate clips (0000, 0001, ...) --------
    for clip in "${LQ_NATIVE}"/*; do
        clip_name=$(basename "${clip}")

        LQ_CLIP="${LQ_NATIVE}/${clip_name}"
        HQ_CLIP="${HQ_NATIVE}/${clip_name}"

        if [[ ! -d "${HQ_CLIP}" ]]; then
            echo "Skipping clip ${clip_name} (HQ missing)"
            continue
        fi

        # -------- iterate frames --------
        for lq_img in "${LQ_CLIP}"/*.png; do
            fname=$(basename "${lq_img}")

            hq_img="${HQ_CLIP}/${fname}"

            if [[ ! -f "${hq_img}" ]]; then
                echo "Missing HQ frame: ${hq_img}"
                continue
            fi

            out_name="${scene_name}_${clip_name}_${fname}"

            cp "${lq_img}" "${LQ_DIR}/${out_name}"
            cp "${hq_img}" "${HQ_DIR}/${out_name}"
        done
    done
done
echo "Done."
