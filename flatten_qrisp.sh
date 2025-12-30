#!/bin/bash
set -e

# ================= DEFAULTS =================

SRC_ROOT="QRISP/TestSet"

LQ_RES="270p"
HQ_RES="1080p"

LQ_TYPE="Native"      # Native | MipBiasMinusX | MipBiasMinusXJittered
HQ_TYPE="Native"      # Native | Enhanced

# ================= ARGUMENT PARSING =================

usage() {
    echo "Usage:"
    echo "  $0 [options]"
    echo
    echo "Options:"
    echo "  --lq-res <270p|370p|540p>"
    echo "  --hq-res <1080p>"
    echo "  --lq-type <Native|MipBiasMinusX|MipBiasMinusXJittered>"
    echo "  --hq-type <Native|Enhanced>"
    echo
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lq-res)   LQ_RES="$2"; shift 2 ;;
        --hq-res)   HQ_RES="$2"; shift 2 ;;
        --lq-type)  LQ_TYPE="$2"; shift 2 ;;
        --hq-type)  HQ_TYPE="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ================= MIP BIAS DERIVATION =================

case "${LQ_RES}" in
    270p) MIP_BIAS="Minus2" ;;
    370p) MIP_BIAS="Minus1.58" ;;
    540p) MIP_BIAS="Minus1" ;;
    *)
        echo "Invalid LQ_RES: ${LQ_RES}"
        exit 1
        ;;
esac

# ================= PATH MAPPING =================

# ---- LQ ----
case "${LQ_TYPE}" in
    Native)
        LQ_SUBDIR="Native"
        LQ_TAG="Native"
        ;;
    MipBiasMinusX)
        LQ_SUBDIR="MipBias${MIP_BIAS}"
        LQ_TAG="MipBias_${MIP_BIAS}"
        ;;
    MipBiasMinusXJittered)
        LQ_SUBDIR="MipBias${MIP_BIAS}Jittered"
        LQ_TAG="MipBias_${MIP_BIAS}_Jittered"
        ;;
    *)
        echo "Invalid LQ_TYPE: ${LQ_TYPE}"
        exit 1
        ;;
esac

# ---- HQ ----
case "${HQ_TYPE}" in
    Native)
        HQ_SUBDIR="Native"
        HQ_TAG="Native"
        ;;
    Enhanced)
        HQ_SUBDIR="Enhanced"
        HQ_TAG="Enhanced"
        ;;
    *)
        echo "Invalid HQ_TYPE: ${HQ_TYPE}"
        exit 1
        ;;
esac

# ================= OUTPUT NAME =================

DST_ROOT="QRISP_${LQ_RES}_${LQ_TAG}_TO_${HQ_RES}_${HQ_TAG}"

echo "=============================================="
echo "Source: ${LQ_RES} / ${LQ_TAG}"
echo "Target: ${HQ_RES} / ${HQ_TAG}"
echo "Output: ${DST_ROOT}"
echo "=============================================="
echo

# ================= SCRIPT =================

for scene in "${SRC_ROOT}"/*; do
    scene_name=$(basename "${scene}")
    echo "Processing scene: ${scene_name}"

    LQ_SCENE="${scene}/${LQ_RES}/${LQ_SUBDIR}"
    HQ_SCENE="${scene}/${HQ_RES}/${HQ_SUBDIR}"

    if [[ ! -d "${LQ_SCENE}" || ! -d "${HQ_SCENE}" ]]; then
        echo "  Skipping scene (missing LQ or HQ folders)"
        continue
    fi

    OUT_SCENE="${DST_ROOT}/${scene_name}"
    OUT_LQ="${OUT_SCENE}/lq"
    OUT_HQ="${OUT_SCENE}/hq"
    mkdir -p "${OUT_LQ}" "${OUT_HQ}"

    for seq in "${LQ_SCENE}"/*; do
        seq_name=$(basename "${seq}")

        LQ_SEQ="${LQ_SCENE}/${seq_name}"
        HQ_SEQ="${HQ_SCENE}/${seq_name}"

        if [[ ! -d "${HQ_SEQ}" ]]; then
            echo "  Skipping sequence ${seq_name} (HQ missing)"
            continue
        fi

        for lq_img in "${LQ_SEQ}"/*.png; do
            fname=$(basename "${lq_img}")
            hq_img="${HQ_SEQ}/${fname}"

            [[ -f "${hq_img}" ]] || continue

            cp "${lq_img}" "${OUT_LQ}/${seq_name}_${fname}"
            cp "${hq_img}" "${OUT_HQ}/${seq_name}_${fname}"
        done
    done
done

echo "Done."
